import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def mse(pred, gt):
    return F.mse_loss(pred,gt).item()

def psnr(pred, gt):
    mse_val = F.mse_loss(pred,gt)
    return 20 * torch.log10(1.0/torch.sqrt(mse_val)).item()

def ssim_metric(pred, gt):
    return ssim(pred,gt,data_range=1.0).item()

import torch
import torch.nn.functional as F


def _uicm(img):
    r, g, b = img[:,0], img[:,1], img[:,2]

    rg = r - g
    yb = 0.5*(r + g) - b

    mu_rg = torch.mean(rg, dim=[1,2])
    mu_yb = torch.mean(yb, dim=[1,2])

    sigma_rg = torch.std(rg, dim=[1,2])
    sigma_yb = torch.std(yb, dim=[1,2])

    return -0.0268*mu_rg + 0.1586*sigma_rg + 0.0733*sigma_yb


def _uism(img):
    gray = img.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                           dtype=img.dtype, device=img.device).view(1,1,3,3)

    grad = F.conv2d(gray, sobel_x, padding=1)
    return torch.mean(torch.abs(grad), dim=[1,2,3])


def _uiconm(img):
    max_val = torch.amax(img, dim=[1,2,3])
    min_val = torch.amin(img, dim=[1,2,3])
    return (max_val - min_val) / (max_val + min_val + 1e-6)


def uiqm(img):
    img = torch.clamp(img, 0, 1)

    uicm = _uicm(img)
    uism = _uism(img)
    uiconm = _uiconm(img)

    return torch.mean(0.0282*uicm + 0.2953*uism + 3.5753*uiconm).item()

def uciqe(img):
    img = torch.clamp(img, 0, 1)

    r, g, b = img[:,0], img[:,1], img[:,2]

    # chroma
    chroma = torch.sqrt((r - g)**2 + (r - b)**2 + (g - b)**2)

    sigma_c = torch.std(chroma, dim=[1,2])

    # luminance
    lum = 0.299*r + 0.587*g + 0.114*b
    con_l = torch.std(lum, dim=[1,2])

    # saturation
    sat = torch.std(img, dim=[1,2,3])

    return torch.mean(0.4680*sigma_c + 0.2745*con_l + 0.2576*sat).item()
