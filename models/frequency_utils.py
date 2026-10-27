# models/frequency_utils.py

import torch
import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)

# models/frequency_utils.py

import torch
import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)

def log_if_main(msg: str):
    # chỉ log nếu file này được gọi trực tiếp
    if __name__ == "__main__":
        logger.info(msg)

def global_frequency_masking(fft, mask_ratio=0.15, scale=3,
                              target_band='all', target='Full-image', dim_size=None):
    """
    Apply random masking to a specific frequency band in the FFT-shifted frequency domain.
    """
    assert target_band in ['low', 'mid', 'high', 'all']
    assert target in ['Full-image', 'Channel']
    assert dim_size is not None
    assert 0 <= mask_ratio <= 1
    assert scale > 2

    B, C = fft.shape[:2]

    if target == 'Full-image':
        H = W = dim_size
        cx, cy = H // 2, W // 2

        y = torch.arange(H, device=fft.device).view(-1, 1).repeat(1, W)
        x = torch.arange(W, device=fft.device).view(1, -1).repeat(H, 1)
        dist_x = torch.abs(x - cx)
        dist_y = torch.abs(y - cy)

        inner = H // (scale * 2)
        middle = H // scale
        outer = H // (scale // 2)

        if target_band == 'low':
            band_mask = (dist_x <= inner) & (dist_y <= inner)
        elif target_band == 'mid':
            band_mask = ((dist_x <= middle) & (dist_y <= middle)) & ~((dist_x <= inner) & (dist_y <= inner))
        elif target_band == 'high':
            band_mask = ((dist_x <= outer) & (dist_y <= outer)) & ~((dist_x <= middle) & (dist_y <= middle))
        else:
            band_mask = torch.ones((H, W), dtype=torch.bool, device=fft.device)

        total = band_mask.sum().item()
        num_mask = int(mask_ratio * total)

        idx = torch.nonzero(band_mask.flatten(), as_tuple=False)
        selected = idx[torch.randperm(idx.size(0))[:num_mask]]

        final_mask = torch.ones(H * W, dtype=torch.bool, device=fft.device)
        final_mask[selected] = False
        final_mask = final_mask.view(H, W)

        fft_masked = fft.clone()
        for b in range(B):
            for c in range(C):
                fft_masked[b, c][~final_mask] = 0

    elif target == 'Channel':
        C = dim_size
        c_mid = C // 2
        inner = C // (scale * 2)
        middle = C // scale
        outer = C // (scale // 2)

        mask = torch.zeros(C, dtype=torch.bool, device=fft.device)
        if target_band == 'low':
            mask[c_mid - inner: c_mid + inner] = True
        elif target_band == 'mid':
            mask[c_mid - middle: c_mid + middle] = True
            mask[c_mid - inner: c_mid + inner] = False
        elif target_band == 'high':
            mask[c_mid - outer: c_mid + outer] = True
            mask[c_mid - middle: c_mid + middle] = False
        else:
            mask[:] = True

        total = mask.sum().item()
        num_mask = int(mask_ratio * total)

        idx = torch.nonzero(mask, as_tuple=False)
        selected = idx[torch.randperm(idx.size(0))[:num_mask]]

        final_mask = torch.ones(C, dtype=torch.bool, device=fft.device)
        final_mask[selected] = False

        fft_masked = fft.clone()
        for b in range(B):
            fft_masked[b, ~final_mask, :, :] = 0

    log_if_main(f"Masked {num_mask}/{total} frequency elements in '{target_band}' band ({target}).")
    
    return fft_masked

def frequency_suppression(fft, target_band='low', scale=3, target='Full-image'):
    """
    Completely suppress a frequency band in the FFT-shifted frequency domain.
    """
    assert target_band in ['low', 'low+mid']
    assert target in ['Full-image', 'Channel']
    assert scale > 2

    fft_masked = fft.clone()
    B, C = fft.shape[:2]

    if target == 'Full-image':
        H = W = fft.shape[-1]
        cx, cy = H // 2, W // 2

        y = torch.arange(H, device=fft.device).view(-1, 1).repeat(1, W)
        x = torch.arange(W, device=fft.device).view(1, -1).repeat(H, 1)
        dist_x = torch.abs(x - cx)
        dist_y = torch.abs(y - cy)

        inner = H // (scale * 2)
        middle = H // scale

        mask = torch.zeros((H, W), dtype=torch.bool, device=fft.device)
        if 'low' in target_band:
            mask |= (dist_x <= inner) & (dist_y <= inner)
        if 'mid' in target_band:
            mid_mask = ((dist_x <= middle) & (dist_y <= middle)) & ~((dist_x <= inner) & (dist_y <= inner))
            mask |= mid_mask

        for b in range(B):
            for c in range(C):
                fft_masked[b, c][mask] = 0

    elif target == 'Channel':
        c_mid = C // 2
        inner = C // (scale * 2)
        middle = C // scale

        mask = torch.zeros(C, dtype=torch.bool, device=fft.device)
        if 'low' in target_band:
            mask[c_mid - inner: c_mid + inner] = True
        if 'mid' in target_band:
            mid_mask = torch.zeros(C, dtype=torch.bool, device=fft.device)
            mid_mask[c_mid - middle: c_mid + middle] = True
            mid_mask[c_mid - inner: c_mid + inner] = False
            mask |= mid_mask

        for b in range(B):
            fft_masked[b, mask, :, :] = 0

    log_if_main(f"Suppressed '{target_band}' frequencies ({target}).")

    return fft_masked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply frequency masking or suppression to input tensor.")
    parser.add_argument('--mode', type=str, choices=['mask', 'zero'], required=True,
                        help='Mode: "mask" for random masking, "zero" for suppression')
    parser.add_argument('-t', '--target', type=str, choices=['Full-image', 'Channel'], default='Full-image',
                        help='Apply operation on Full-image or Channel domain')
    parser.add_argument('-b', '--band', type=str, default='high',
                        help='Target frequency band: "low", "mid", "high", "all", or "low+mid"')
    parser.add_argument('--shape', type=int, default=64, help='Input image size (assumes square)')
    parser.add_argument('-m', '--mask_ratio', type=float, default=0.2,
                        help='Ratio of masking (only used if mode == "mask")')
    parser.add_argument('-s', '--scale', type=int, default=3, help='Scaling factor for frequency region definition')
    args = parser.parse_args()

    dummy_input = torch.randn(2, 3, args.shape, args.shape)

    fft = torch.fft.fft2(dummy_input, dim=(-2, -1), norm='ortho')
    fft = torch.fft.fftshift(fft, dim=(-2, -1))

    if args.mode == 'mask':
        result = global_frequency_masking(
            fft,
            mask_ratio=args.mask_ratio,
            target_band=args.band,
            target=args.target,
            dim_size=args.shape,
            scale=args.scale
        )
    elif args.mode == 'zero':
        result = frequency_suppression(
            fft,
            target_band=args.band,
            target=args.target,
            scale=args.scale
        )

    logger.info("CLI operation completed successfully.")