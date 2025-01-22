import os
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from timm import create_model
import torch
from torchvision import models
import torchvision.transforms as transforms

from utils.utils import draw_random_path


def draw_path_on_image(image, coordinates, path_color, box_color):
    # Draw the path
    for pixel in coordinates:
        image[pixel[0],pixel[1]] = path_color

    # Draw bounding box
    rows, cols = zip(*coordinates)
    top_left = (max(0, min(cols) - 5), max(0, min(rows) - 5))
    bottom_right = (min(image.shape[1] - 1, max(cols) + 5), min(image.shape[0] - 1, max(rows) + 5))
    cv2.rectangle(image, top_left, bottom_right, box_color, 2)

def evaluate_and_visualize(psp_folder, base_dir, hologram_mask_path, model, transform, class_names, device, json_path, path_colors, output_dir="visualization"):
    """
    Evaluates images in multiple folders, makes predictions, and visualizes random paths based on predictions.

    Parameters:
        psp_folder: folder names to evaluate.
        base_dir: The base directory containing the folders.
        hologram_mask_path: Path to the hologram mask image.
        model: The trained PyTorch model for evaluation.
        transform: The image transformation pipeline.
        class_names: List of class names
        device: Device to run the evaluation ("cuda" or "cpu").
        json_path: Path to the JSON file containing random paths.
        path_colors : Dictionary specifying colors for paths, e.g., {"origins": (0, 255, 0), "fraud": (255, 0, 0)}.
        output_dir: Directory to save visualized images. Default is "visualization".
    """
    # Ensure output directory exists
    type_folder = os.path.basename(base_dir)
    os.makedirs(f"{output_dir}/{type_folder}", exist_ok=True)

    # Load JSON data for random paths
    with open(json_path, 'r') as f:
        data = json.load(f)

    test_dir = os.path.join(base_dir, psp_folder)
    image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))]

    # Lists to store image names for each category
    origins_list = []
    fraud_list = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)  # Extract the name of the image

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_label = class_names[predicted_class_idx]

        # Draw paths on the hologram mask based on prediction
        if predicted_label == "origins":
            origins_list.append(image_name)
        else:
            fraud_list.append(image_name)
    
    print(f"Number of 'origins' images: {len(origins_list)}")
    print(f"Number of 'fraud' images: {len(fraud_list)}")
        
    # Visualize random paths for each category
    img = cv2.imread(hologram_mask_path)
    for image_name in origins_list:
        filtered_data = {path: coords for path, coords in data.items() if image_name in path}
        list_coordinates = list(filtered_data.values())

        for coordinates in list_coordinates:
            draw_path_on_image(img, coordinates, path_color=path_colors["origins"], box_color=(0, 255, 0))
    
    for image_name in fraud_list:
        filtered_data = {path: coords for path, coords in data.items() if image_name in path}
        list_coordinates = list(filtered_data.values())

        for coordinates in list_coordinates:
            draw_path_on_image(img, coordinates, path_color=path_colors["fraud"], box_color=(0, 255, 0))
    
    cv2.imwrite(f"{output_dir}/{type_folder}/{psp_folder}_predictions.jpg", img)

    plt.imshow(img, cmap='gray')
    plt.show()



if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define class names
    class_names = ['fraud', 'origins']
    model_path = "best_model/best_model_boostrap.pth"
    # Define image transformation pipeline
    
    if "vit" in model_path:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        model = create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load saved weights
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load the trained model
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # Define paths and folders
    base_dir = "D:/MIDV/str_images_test/evaluate_copy_without_holo"
    psp_folder= "psp09_04_01"
    hologram_mask_path = "D:/MIDV/templates/hologram_masks/passport_hologram_mask_small.png"
    json_path = "D:/MIDV/str_images_test/evaluate_copy_without_holo/coordinates.json"
    output_dir = "visualization"

    # Evaluate and visualize
    path_colors={"origins": (255, 0, 0), "fraud": (0, 0, 255)}
    evaluate_and_visualize(psp_folder, base_dir, hologram_mask_path, model, transform, class_names, device, json_path, path_colors, output_dir)
