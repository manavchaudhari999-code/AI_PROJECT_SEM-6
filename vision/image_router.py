import clip
import torch
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

LABELS = [
    "chest x-ray radiograph",
    "lung x-ray image",
    "thoracic radiograph",
    "medical radiology x-ray",
    "bone x-ray",
    "hand x-ray",
    "leg x-ray",
    "skull x-ray",
    "medical grayscale radiograph",
    "skin lesion photograph",
    "eye fundus photograph",
    "normal everyday photograph"
]


def detect_image_type(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(LABELS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits = (image_features @ text_features.T).softmax(dim=-1)

    best_index = logits.argmax().item()
    confidence = float(logits[0][best_index])

    if confidence < 0.25:
        return "uncertain medical image", confidence

    return LABELS[best_index], confidence
