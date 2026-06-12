import os, cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, gt_dir, size=256):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.files = os.listdir(input_dir)
        self.size = size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size,size))
        ])

    def find_gt(self, name):
        base = os.path.splitext(name)[0]

        # try jpg png jpeg
        for ext in [".jpg",".png",".jpeg",".JPG",".PNG"]:
            gt_path = os.path.join(self.gt_dir, base+ext)
            if os.path.exists(gt_path):
                return gt_path

        return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        inp_path = os.path.join(self.input_dir,img_name)
        gt_path  = self.find_gt(img_name)

        if gt_path is None:
            # skip if GT missing
            return self.__getitem__((idx+1) % len(self.files))

        inp = cv2.imread(inp_path)
        gt  = cv2.imread(gt_path)

        if inp is None or gt is None:
            return self.__getitem__((idx+1) % len(self.files))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        gt  = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        inp = self.transform(inp)
        gt  = self.transform(gt)

        return inp, gt
