from ultralytics import YOLO
import torch
import os

def train_powerline_model(config_path, pretrained_model='yolov8n.pt', epochs=20):
    """
    Train YOLOv8 model for power line detection 

    Args:
        config_path (str): Path to data.yaml for the dataset
        pretrained_model (str): Pretrained model weight (e.g., 'yolov8n.pt')
        epochs (int): Number of training epochs
    """

    # Load model
    model = YOLO(pretrained_model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Define training arguments. args which are not mentioned would take default args from pretrained model.
    training_args = {
        'data': config_path,
        'imgsz': 640, # default
        'epochs': epochs, # default: 100
        'batch': 16, # default: Auto
        'device': device,
        'patience': 10, # default: 50 . Early stopping sooner: Training stops if no improvement in 10 epochs, speeding up experiments
        'seed': 42, # helps get same results across runs.

        # Augmentations
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.2, # Default: 0.5 .Less aggressive zoom — more stability in object size; better if object scale doesn't vary much. For improving precision on thin/small objects
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.1, # Default: 1 .Less mosaic augmentation — may reduce noise/artifacts but less variety in data.
        'mixup': 0.0,
        'copy_paste': 0.0,

        # Learning rate scheduler
        'cos_lr': True,

        # Output directory
        'project': 'models',                 # Custom folder for output (e.g., models, weights, logs). Keeps workspace organized.
        'name': 'yolov8n_powerlines',        # Custom run name — makes it easy to identify your experiments.
        'exist_ok': True                     # Allows reuse of the same output folder (doesn't crash if it exists). Helpful for iteration, but may overwrite older runs.
    }

    # Start training
    try:
        results = model.train(**training_args)
        print("Training completed successfully!")
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        return None


if __name__ == '__main__':
    CONFIG_PATH = r"electric_asset\data.yaml"

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing dataset config file at {CONFIG_PATH}")

    results = train_powerline_model(
        config_path=CONFIG_PATH,
        pretrained_model=r"electric_asset\yolov8n.pt",
        epochs=20
    )



# When cos_lr=True:
    # The learning rate follows a cosine curve instead of a straight line
    # It decreases quickly at first, then slows down as it gets closer to the end of training
    # Smoother convergence: Reduces the chance of overshooting or bouncing near minima.
    # Early exploratory phase, late fine-tuning: Starts with larger steps (exploring), ends with smaller ones (refining).
