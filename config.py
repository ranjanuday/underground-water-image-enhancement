import torch

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# paths
UIEB_RAW = '/media/cvblns/NS/Praveen/UW_Datasets/UIEB/raw-890'
UIEB_REF = '/media/cvblns/NS/Praveen/UW_Datasets/UIEB/reference-890'

# EUVP_TRAIN_A = "datasets/EUVP/trainA"
# EUVP_TRAIN_B = "datasets/EUVP/trainB"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100
IMG_SIZE = 256

SAVE_DIR = "checkpoints/best_model.pth"

