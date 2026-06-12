import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import UIE_Net
from dataset import UnderwaterDataset
from metrics import psnr,ssim_metric,mse
# from utils import calculate_metrics
import config
import csv

import os

# create directory safely
os.makedirs("checkpoints", exist_ok=True)

device = config.DEVICE

# datasets
uieb = UnderwaterDataset(config.UIEB_RAW, config.UIEB_REF)
# euvp = UnderwaterDataset(config.EUVP_TRAIN_A, config.EUVP_TRAIN_B)

# dataset = ConcatDataset([uieb,euvp])
dataset = uieb
#loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)
train_size = 800
val_size = 90

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


model = UIE_Net().to(device)
opt = torch.optim.Adam(model.parameters(),lr=config.LR)
loss_fn = torch.nn.L1Loss()
csv_filename = 'training_log.csv'
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'PSNR', 'SSIM', 'MSE'])

best_psnr = 0

for epoch in range(config.EPOCHS):
    model.train()
    loop = tqdm(train_loader)
    running_loss = 0.0
    
    for inp,gt in loop:
        inp,gt = inp.to(device),gt.to(device)

        out = model(inp)
        loss = loss_fn(out,gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ---------- evaluation ----------
    model.eval()
    with torch.no_grad():
        p_total=0
        s_total=0
        m_total=0
        count=0
        val_loss = 0.0
        
        for inp,gt in val_loader:
            inp,gt = inp.to(device),gt.to(device)
            out = model(inp)
            loss = loss_fn(out, gt)
            val_loss += loss.item()
            # p,s,m,_=calculate_metrics(out, gt)
            p_total+=psnr(out,gt)
            s_total+=ssim_metric(out,gt)
            m_total+=mse(out,gt)
            count+=1
            
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr=p_total/count
        avg_ssim=s_total/count
        avg_mse=m_total/count

        print(f"\nPSNR:{avg_psnr:.3f} SSIM:{avg_ssim:.3f}")
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_psnr, avg_ssim, avg_mse])

        # save best
        if avg_psnr>best_psnr:
            best_psnr=avg_psnr
            torch.save(model.state_dict(),config.SAVE_DIR)
            print("🔥 Best model saved")
