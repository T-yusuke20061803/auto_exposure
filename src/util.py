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
    # 現在の画像の明るさ(lum)を計算
    lum = get_luminance(hdr)
    # 平均輝度が「0.18 (18%グレー)」になるために必要な補正量(stop_value)を計算
    stop_value = estimate_exposure_compensation_by_mean_luminance(lum, middle_gray)
    # 画像を補正して、強制的に「見やすい明るさ」にする
    hdr = exposure_compensation(hdr, stop_value)
    #(確認用) 補正後の輝度を再計算
    lum = get_luminance(hdr)
    # カメラの露出状態を記録
    # 「画像を+3.0段明るくした」なら、元のカメラ設定は「-3.0段暗かった」という意味
    camera_ev = -stop_value
    # 正解ラベル(ev)の更新 ★ここが重要★
    # 画像を自動で明るくしてしまった分、AIが予測すべき「残りの補正量」は減る
    # 新ラベル = 元ラベル - 自動補正量
    given_ev = ev - stop_value
    return hdr, given_ev, camera_ev


def get_luminance(rgb):
    """
    Args:
        rgb: ndarray of shape (..., 3)
    """
    return np.dot(rgb, [0.2126, 0.7152, 0.0722])
