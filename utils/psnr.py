import PIL
import math
import numpy as np
import skimage.color as sc

def PSNR(img1, img2, ycbcr=False, shave=0):
    if ycbcr:
        a = np.float32(img1)
        b = np.float32(img2)
        a = sc.rgb2ycbcr(a / 255)[:, :, 0]
        b = sc.rgb2ycbcr(b / 255)[:, :, 0]
    else:
        a = np.array(img1).astype(np.float32)
        b = np.array(img2).astype(np.float32)
        
    if shave:
        a = a[shave:-shave, shave:-shave]
        b = b[shave:-shave, shave:-shave]
    
    mse = np.mean((a - b) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return np.minimum(100.0, 20 * np.math.log10(PIXEL_MAX) - 10 * np.math.log10(mse))