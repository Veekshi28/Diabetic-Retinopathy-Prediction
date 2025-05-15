import os
import json
import base64
import logging
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
        # Initialize model and transform once
        if model is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'best_dr_model.pth'))
            logging.info(f"Loading model from: {model_path}")

            model = build_model()
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            model.half()  # Reduce memory footprint
            transform = get_transform()

            logging.info("Model loaded & transform initialized.")

        # Parse incoming request body
        try:
            req_body = req.get_json()
        except Exception as e:
            logging.error(f"Failed to parse JSON: {str(e)}")
            return func.HttpResponse("Invalid JSON body", status_code=400)

        logging.info(f"Received request: {req_body}")

        img_base64 = req_body.get('image')
        if img_base64 is None:
            return func.HttpResponse("Missing 'image' key in request body", status_code=400)

        # Handle potential data URL prefix and padding
        img_base64 = img_base64.split(",")[-1]
        if len(img_base64) % 4 != 0:
            img_base64 += '=' * (4 - len(img_base64) % 4)

        # Decode base64 image
        img_bytes = base64.b64decode(img_base64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        logging.info(f"Image buffer decoded. Buffer size: {np_img.shape}")

        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return func.HttpResponse("Failed to decode image", status_code=400)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logging.info(f"Image converted to RGB: shape={img.shape}")

        # Apply albumentations transform
        transformed = transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0).half()  # Match model precision
        logging.info(f"Transformed image tensor shape: {img_tensor.shape}")

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = outputs.argmax(1).item()
            logging.info(f"Model predicted index: {pred_idx}")

        # Map to class name
        class_map = {
            0: 'No_DR',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Proliferative_DR'
        }
        prediction = class_map.get(pred_idx, "Unknown")

        response = {"prediction": prediction}
        logging.info(f"Response: {response}")

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Internal Server Error: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
