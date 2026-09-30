import cv2
import numpy as np
from typing import Any

def apply_synchronous_shear(image, shear_factor, dem_mask: None | Any = None, seg_mask: None | Any = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if shear_factor == 0:
        return image, dem_mask, seg_mask
    
    h, w = image.shape[:2]

    M = np.array([[1, shear_factor, -shear_factor * ((h-1) / 2)], 
                  [0, 1, 0]], dtype=np.float32)
    
    sh_image = cv2.warpAffine(image, M,(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    if dem_mask is not None:
        sh_dem = cv2.warpAffine(dem_mask, M,(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return sh_image, sh_dem, None
    elif seg_mask is not None:
        sh_seg = cv2.warpAffine(seg_mask, M,(w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        return sh_image, None, sh_seg
    

