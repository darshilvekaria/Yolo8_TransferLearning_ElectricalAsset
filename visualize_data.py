import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_yolo_labels(image_path, label_path, class_names=None):
    """
    Visualize YOLO labels (bounding boxes or polygons) on the image.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO label .txt file.
        class_names (list, optional): List of class names. If None, class IDs will be used.
    """
    # Load and prepare the image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
        return

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # Read and parse labels
    with open(label_path, 'r') as f:
        lines = f.read().strip().split('\n')

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # skip invalid lines

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Bounding box format
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)

            cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            label = class_names[class_id] if class_names else str(class_id)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Polygon format
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])

            pts_array = np.array([points], np.int32)
            cv2.polylines(image, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

            label = class_names[class_id] if class_names else str(class_id)
            x0, y0 = points[0]
            cv2.putText(image, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Visualizing: {os.path.basename(image_path)}")
    plt.show()


# -------- Example usage --------

if __name__ == "__main__":
    # Replace these paths with your actual image and label path

    image_path = r"electric_asset\dataset\train\images\1-3-_jpg.rf.51ef721db872f9163041283c903ed305.jpg"
    label_path = r"electric_asset\dataset\train\labels\1-3-_jpg.rf.51ef721db872f9163041283c903ed305.txt"

    # Replace or modify class names as per your dataset
    class_names = ['Broken Cable', 'Broken Insulator', 'Cable', 'Insulators', 'Tower', 'Vegetation', 'object']

    visualize_yolo_labels(image_path, label_path, class_names)
