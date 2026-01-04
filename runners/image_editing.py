import os
import glob
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from models.diffusion import Model
from functions.process_data import *


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def image_editing_sample(self):
        print("Loading model")
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "CT_Medical":
            # For medical images, load checkpoint from local path
            # The checkpoint path should be specified in config or args
            ckpt_from_args = getattr(self.args, 'ckpt', None)
            ckpt_from_config = getattr(self.config.data, 'ckpt_path', None)
            ckpt_path = ckpt_from_args if ckpt_from_args else ckpt_from_config
            if ckpt_path is None:
                raise ValueError("For CT_Medical dataset, please specify checkpoint path via --ckpt argument or ckpt_path in config")
            url = None
        else:
            raise ValueError(f"Unknown dataset: {self.config.data.dataset}")

        model = Model(self.config)
        if self.config.data.dataset == "CT_Medical":
            # Load from local checkpoint
            # Note: weights_only=False is required for DDIM checkpoints which may contain
            # non-tensor objects like EMA state. Only load from trusted sources.
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            # Debug: print checkpoint type and keys with flush to ensure output is visible
            import sys
            ckpt_type = str(type(ckpt))
            print(f"Checkpoint type: {ckpt_type}", flush=True)
            
            if hasattr(ckpt, 'keys'):
                try:
                    keys = list(ckpt.keys())
                    print(f"Checkpoint keys: {keys[:10]}{'...' if len(keys) > 10 else ''}", flush=True)
                except Exception as e:
                    print(f"Could not list keys: {e}", flush=True)
                    keys = []
            else:
                keys = []
                print("Checkpoint has no 'keys' attribute", flush=True)
            
            # Handle different checkpoint formats from DDIM and other frameworks
            state_dict = None
            
            # Check if it's a dict-like object (dict, OrderedDict, etc.)
            if hasattr(ckpt, 'keys') and hasattr(ckpt, '__getitem__'):
                # Try common keys used by different frameworks
                if 'model' in keys:
                    state_dict = ckpt['model']
                    print("Using ckpt['model']", flush=True)
                elif 'state_dict' in keys:
                    state_dict = ckpt['state_dict']
                    print("Using ckpt['state_dict']", flush=True)
                elif 'ema' in keys:
                    # DDIM often uses EMA weights
                    state_dict = ckpt['ema']
                    print("Using ckpt['ema']", flush=True)
                elif 'model_state_dict' in keys:
                    state_dict = ckpt['model_state_dict']
                    print("Using ckpt['model_state_dict']", flush=True)
                elif len(keys) > 0:
                    # Check if the first value is a tensor
                    first_key = keys[0]
                    try:
                        first_val = ckpt[first_key]
                        if isinstance(first_val, torch.Tensor):
                            # The checkpoint itself is the state_dict
                            state_dict = ckpt
                            print("Using checkpoint directly as state_dict", flush=True)
                        elif isinstance(first_val, dict):
                            # Try to find state_dict in nested structure
                            for key in keys:
                                val = ckpt[key]
                                if isinstance(val, dict) and len(val) > 0:
                                    subkeys = list(val.keys())
                                    if len(subkeys) > 0 and isinstance(val[subkeys[0]], torch.Tensor):
                                        state_dict = val
                                        print(f"Using ckpt['{key}'] as state_dict", flush=True)
                                        break
                    except Exception as e:
                        print(f"Error accessing checkpoint values: {e}", flush=True)
                
                if state_dict is None:
                    # Last resort: try using the checkpoint directly
                    state_dict = ckpt
                    print("Fallback: using checkpoint directly", flush=True)
            else:
                # Not a dict-like object - this is unexpected
                print(f"ERROR: Checkpoint is not dict-like. Type: {ckpt_type}", flush=True)
                raise TypeError(f"Unexpected checkpoint format. Expected dict-like object, got {ckpt_type}. "
                               f"Please check the checkpoint file: {ckpt_path}")
            
            # Handle potential 'module.' prefix from DataParallel
            if state_dict is not None and hasattr(state_dict, 'items'):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            
            print(f"Loading state_dict with {len(state_dict) if state_dict else 0} keys", flush=True)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded checkpoint from {ckpt_path}", flush=True)
        else:
            ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0

        n = self.config.sampling.batch_size
        model.eval()
        print("Start sampling")
        
        # Determine input files to process
        input_path = self.args.npy_name
        input_files = []
        
        if os.path.isdir(input_path):
            # Directory mode: process all .pth files in the directory
            pth_files = sorted(glob.glob(os.path.join(input_path, "*.pth")))
            if len(pth_files) == 0:
                raise FileNotFoundError(f"No .pth files found in directory: {input_path}")
            input_files = pth_files
            print(f"Found {len(input_files)} .pth files in {input_path}")
        elif os.path.isfile(input_path):
            # Direct file path
            input_files = [input_path]
        elif os.path.isfile(input_path + ".pth"):
            # File path without extension
            input_files = [input_path + ".pth"]
        elif os.path.isfile(os.path.join("colab_demo", input_path + ".pth")):
            # Original format: name only, look in colab_demo/
            download_process_data(path="colab_demo")
            input_files = [os.path.join("colab_demo", input_path + ".pth")]
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")
        
        with torch.no_grad():
            for file_idx, input_file in enumerate(input_files):
                print(f"\nProcessing [{file_idx+1}/{len(input_files)}]: {input_file}")
                
                # Get base name for output files
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                
                # Note: weights_only=False is required for SDEdit input format [mask, img]
                # Only load from trusted sources.
                [mask, img] = torch.load(input_file, weights_only=False)

                mask = mask.to(self.config.device)
                img = img.to(self.config.device)
                img = img.unsqueeze(dim=0)
                img = img.repeat(n, 1, 1, 1)
                x0 = img

                tvu.save_image(x0, os.path.join(self.args.image_folder, f'{base_name}_original_input.png'))
                x0 = (x0 - 0.5) * 2.

                for it in range(self.args.sample_step):
                    e = torch.randn_like(x0)
                    total_noise_levels = self.args.t
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'{base_name}_init_{it}.png'))

                    with tqdm(total=total_noise_levels, desc=f"{base_name} Iteration {it}") as progress_bar:
                        for i in reversed(range(total_noise_levels)):
                            t = (torch.ones(n) * i).to(self.device)
                            x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                            logvar=self.logvar,
                                                                            betas=self.betas)
                            x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                            x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                            # added intermediate step vis
                            if (i - 99) % 100 == 0:
                                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'{base_name}_noise_t_{i}_{it}.png'))
                            progress_bar.update(1)

                    x0[:, (mask != 1.)] = x[:, (mask != 1.)]
                    torch.save(x, os.path.join(self.args.image_folder,
                                               f'{base_name}_samples_{it}.pth'))
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f'{base_name}_samples_{it}.png'))
                
                print(f"Completed: {base_name}")
