import os
import json
import base64
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import azure.functions as func

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

def main(req: func.HttpRequest) -> func.HttpResponse:
    global model, transform

    try:
        # Initialize model and transform only once
        if model is None:
            model_path = os.path.join(os.path.dirname(__file__), 'best_dr_model.pth')
            print(f"Loading model from: {model_path}")
            model = build_model()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            transform = get_transform()
            print("Model loaded and transform initialized.")

        # Parse incoming request
        req_body = req.get_json()
        print(f"Received JSON body: {req_body}")

        img_base64 = req_body.get('image')
        if img_base64 is None:
            raise ValueError("No 'image' key found in request body")

        # Decode base64 image
        img_bytes = base64.b64decode(img_base64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        print(f"Decoded image buffer: shape={np_img.shape}")

        # Decode image and convert to RGB
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed to decode image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image converted to RGB: shape={img.shape}")

        # Apply transformations
        img = transform(image=img)['image'].unsqueeze(0)
        print(f"Transformed image tensor shape: {img.shape}")

        # Inference
        with torch.no_grad():
            outputs = model(img)
            pred_idx = outputs.argmax(1).item()
            print(f"Model prediction index: {pred_idx}")

        # Map prediction to class name
        class_map = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative_DR'}
        prediction = class_map.get(pred_idx, "Unknown")
        print(f"Predicted Class: {prediction}")

        response = {
            "prediction": prediction
        }

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        error_response = {
            "error": str(e)
        }
        return func.HttpResponse(
            body=json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )
