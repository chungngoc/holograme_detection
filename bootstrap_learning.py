import os
import shutil
from PIL import Image
import torch

from torchvision import models, datasets
import torchvision.transforms as transforms

def classify_and_clean(dataset_dir, confused_dir, model, transform, class_names, device, threshold = 0.4):
    origins_path = os.path.join(dataset_dir, "origins")
    image_paths = [os.path.join(origins_path, img) for img in os.listdir(origins_path) if img.endswith(('.jpg', '.png'))]
    count = 0
    for image_path in image_paths:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_label = class_names[predicted_class_idx]

            # Check if misclassified
            origins_prob = probabilities[class_names.index("origins")].item()
            if predicted_label != "origins" and origins_prob < threshold:
                destination_path = os.path.join(confused_dir, os.path.basename(image_path))
                shutil.move(image_path, destination_path)
                count +=1
    print(f"Moved {count} images")

if __name__ == "__main__":
    # Paths
    DATASET_DIR = "D:/MIDV/str_images_symmetric_copy/val"
    CONFUSED_DIR = "D:/MIDV/str_images_symmetric_copy/confused_images"

    # Create directory for misclassified images
    os.makedirs(CONFUSED_DIR, exist_ok=True)

    # Define class names
    class_names = ['fraud', 'origins']
    model_path = "best_model/best_model_1.pth"

    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the trained model
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # Classify and clear dataset
    classify_and_clean(DATASET_DIR, CONFUSED_DIR, model, transform, class_names, device, threshold = 0.4)