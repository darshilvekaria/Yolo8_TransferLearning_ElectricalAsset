from ultralytics import YOLO
import os

def run_inference():
    model_path = 'models/yolov8n_powerlines/weights/best.pt'
    model = YOLO(model_path)
    
    results = model.predict(
        source='dataset/test/images',  # Use test set
        imgsz=640,
        conf=0.25,
        save=True,
        project='outputs',
        name='predictions',
        exist_ok=True
    )
    print("Inference completed. Predictions saved in outputs/predictions")

if __name__ == "__main__":
    run_inference()
