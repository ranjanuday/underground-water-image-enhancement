import torch
import cv2
import os
import torch.nn.functional as F
from torchvision import transforms

from model import UIE_Net
from metrics import psnr, ssim_metric, uiqm, uciqe
import config


device = config.DEVICE

# -------- model --------
model = UIE_Net().to(device)
model.load_state_dict(torch.load(config.SAVE_DIR, map_location=device, weights_only=True))
model.eval()

transform = transforms.ToTensor()

# input_dir = "datasets/EUVP/testA"
# gt_dir = "datasets/EUVP/testB"
input_dir = '/media/cvblns/NS/Praveen/UW_Datasets/UIEB/raw-890'
gt_dir = '/media/cvblns/NS/Praveen/UW_Datasets/UIEB/reference-890'
os.makedirs("outputs", exist_ok=True)

print("Starting inference + evaluation...\n")


# -------- metrics accumulators --------
p_total, s_total, u_total, c_total = 0.0, 0.0, 0.0, 0.0
count = 0

# -------- save control --------
# save_limit = 20
# saved = 0


# -------- loop --------
for img in sorted(os.listdir(input_dir)):

    inp_path = os.path.join(input_dir, img)
    gt_path = os.path.join(gt_dir, img)

    im = cv2.imread(inp_path)
    gt = cv2.imread(gt_path)

    if im is None or gt is None:
        continue

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    inp_tensor = transform(rgb).unsqueeze(0).to(device)
    gt_tensor = transform(gt).unsqueeze(0).to(device)

    # -------- padding (safe) --------
    _, _, h, w = inp_tensor.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4

    if pad_h or pad_w:
        inp_tensor = F.pad(inp_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    # -------- inference --------
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            out_tensor = model(inp_tensor)

    # -------- crop --------
    out_tensor = out_tensor[:, :, :h, :w]

    # -------- save only few --------
    # if saved < save_limit:
    #     out_img = (out_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    #     cv2.imwrite(f"outputs/{img}", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    #     saved += 1
    # if saved < save_limit:
    out_img = (out_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    cv2.imwrite(f"outputs/{img}", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    # saved += 1

    # -------- metrics --------
    try:
        p = float(psnr(out_tensor, gt_tensor))
        s = float(ssim_metric(out_tensor, gt_tensor))
        u = float(uiqm(out_tensor))
        c = float(uciqe(out_tensor))
    except:
        continue

    print(f"{img:25s} | PSNR: {p:.2f} | SSIM: {s:.3f} | UIQM: {u:.3f} | UCIQE: {c:.3f}")

    p_total += p
    s_total += s
    u_total += u
    c_total += c
    count += 1


# -------- final results --------
if count > 0:
    print("\n===== FINAL RESULTS =====")
    print(f"Images evaluated: {count}")
    print(f"Avg PSNR : {p_total / count:.3f}")
    print(f"Avg SSIM : {s_total / count:.3f}")
    print(f"Avg UIQM : {u_total / count:.3f}")
    print(f"Avg UCIQE: {c_total / count:.3f}")
    print("========================")
else:
    print("No valid images processed.")