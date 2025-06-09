import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add MiDaS to Python path
sys.path.append("/home/thurapyaesone/Desktop/nerf2/MiDaS")

from midas.midas_net import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_midas_model(device):
    model_path = "/home/thurapyaesone/Desktop/nerf2/MiDaS/weights/midas_v21_384.pt"

    model = MidasNet_small(
        model_path,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={'expand': True}
    )
    model.eval()
    model.to(device)
    return model

def predict_depth(model, img, device):
    transform = Resize(
        384, 384,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="minimal"
    )

    transformed = transform({"image": img})
    img_input = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed)["image"]
    img_input = PrepareForNet()(img_input)

    img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_input)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()
    return depth

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_midas_model(device)

    input_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/images"
    output_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/midas_depths"
    os.makedirs(output_folder, exist_ok=True)

    img_list = sorted([f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg"))])

    for img_name in tqdm(img_list, desc="Predicting depths"):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = predict_depth(model, img, device)

        np.save(os.path.join(output_folder, img_name.replace(".png", ".npy").replace(".jpg", ".npy")), depth)
