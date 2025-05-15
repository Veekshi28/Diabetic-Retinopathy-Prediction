import os
import json
import base64
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def build_model(num_classes=5):
    model = timm.create_model('resnet50', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

model = None
transform = None

def main(req):
    global model, transform

    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_dr_model.pth')
        model = build_model()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        transform = get_transform()

    try:
        req_body = req.get_json()
        img_base64 = req_body.get('image')

        img_bytes = base64.b64decode(img_base64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = transform(image=img)['image'].unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            pred_idx = outputs.argmax(1).item()

        class_map = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative_DR'}

        return {
            "status": 200,
            "body": {"prediction": class_map[pred_idx]}
        }

    except Exception as e:
        return {
            "status": 500,
            "body": f"Error: {str(e)}"
        }
