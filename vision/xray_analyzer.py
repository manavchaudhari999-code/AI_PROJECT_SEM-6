import torch
import torchxrayvision as xrv
import cv2
import numpy as np

model = xrv.models.DenseNet(weights="densenet121-res224-all")

def analyze_xray(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0,1))

    with torch.no_grad():
        preds = model(torch.tensor(img, dtype=torch.float32))

    pneumonia_score = float(preds[0][model.pathologies.index("Pneumonia")])
    return pneumonia_score
