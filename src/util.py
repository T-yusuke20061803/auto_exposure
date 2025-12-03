from typing import Optional
import numpy as np

import torch



def set_random_seed(seed: Optional[int] = None, is_test: Optional[bool] = None) -> None:
    if seed is not None: 
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if is_test:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def estimate_exposure_compensation_by_mean_luminance(lum: np.ndarray, target_luminance=0.18) -> np.ndarray:
    lum[lum <= 0] = np.sqrt(np.finfo(np.float32).eps)
    log_lum = np.log2(lum)
    mean_luminance = 2 ** np.mean(log_lum)
    stop_value = np.log2(target_luminance / mean_luminance)
    return stop_value


def exposure_compensation(image: np.ndarray, stop_value: float) -> np.ndarray:
    # Calculate the exposure compensation factor
    exposure_factor = 2 ** stop_value
    # Adjust the image exposure based on the exposure compensation factor
    adjusted_image = image * exposure_factor

    return adjusted_image


def normalize_hdr(hdr, ev, middle_gray=0.18):
    lum = get_luminance(hdr)
    stop_value = estimate_exposure_compensation_by_mean_luminance(lum, middle_gray)
    hdr = exposure_compensation(hdr, stop_value)
    lum = get_luminance(hdr)
    camera_ev = -stop_value
    given_ev = ev - stop_value
    return hdr, given_ev, camera_ev


def get_luminance(rgb):
    """
    Args:
        rgb: ndarray of shape (..., 3)
    """
    return np.dot(rgb, [0.2126, 0.7152, 0.0722])
