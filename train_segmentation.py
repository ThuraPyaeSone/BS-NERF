# === train_segmentation.py ===
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# ---- Dataset ----
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if os.path.exists(os.path.join(mask_dir, f))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Failed to load: {img_path} or {mask_path}")

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = TF.to_pil_image(image)
        mask = torch.tensor(mask / 255, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask

# ---- Transforms ----
transform = T.Compose([
    T.Resize((256, 256)),  
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(256, scale=(0.9, 1.0)),
    T.ToTensor()
])

# ---- Dataloading ----
train_dataset = SegmentationDataset("data/Knee/images", "data/Knee/masks", transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# ---- Model ----
model = deeplabv3_resnet50(pretrained=False, num_classes=2)
model = model.cuda()

# ---- Loss Function ----
# Use weighted CrossEntropy to handle class imbalance
class_weights = torch.tensor([0.3, 0.7], device='cuda')
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---- Training ----
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.cuda(), masks.cuda()
        output = model(images)['out']
        loss = criterion(output, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")

    scheduler.step()
    print(f"Epoch {epoch+1} Summary: Total Loss = {total_loss:.4f}\n")

# ---- Save Trained Model ----
torch.save(model.state_dict(), "knee_segment_model.pth")
print("[SAVED] knee_segment_model.pth")