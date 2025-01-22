from PIL import Image
import os

import torch
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_names = ['fraud', 'origins']
transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize to match MobileNet's input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load saved model
model_path = "best_model/best_model_boostrap.pth"

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))  # Load saved weights
model = model.to(device).eval() # Evaluation mode

evaluate_folder = "D:/MIDV/str_images_test"
temps = ["evaluate_origins", "evaluate_photo_holo_copy", "evaluate_pseudo_holo_copy"]
for type_folder in temps:
    folder_path = os.listdir(os.path.join(evaluate_folder, type_folder))
    result_file = f"../results/evaluate_{type_folder}.txt"
    #Create file results
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    # Get list of psp folder, not a json file
    psp_folder_list =[x for x in folder_path if os.path.isdir(os.path.join(evaluate_folder, type_folder, x))] 
    mean_accuracy = 0.0
    for psp_folder in psp_folder_list:
        test_dir = os.path.join(evaluate_folder, type_folder, psp_folder)
        images = os.listdir(test_dir)
        image_paths = [os.path.join(test_dir, img) for img in images if img.endswith(('.jpg', '.png'))]
        count_origins = 0
        count_fraud = 0

        for idx, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Softmax for probabilities
                predicted_class_idx = torch.argmax(probabilities).item()  # Get the index of the highest probability
                predicted_label = class_names[predicted_class_idx]  # Map to class label

                if predicted_label == 'origins':
                    count_origins += 1  # Count correct predictions
                else:
                    count_fraud += 1
                
        mean_accuracy += count_origins / len(image_paths)
        
        print(f"Prediction on {psp_folder}")
        print("Number random path original : ", count_origins)
        print("Number random path fraud : ", count_fraud)

        with open(result_file, "a") as f:
            f.write(f"{psp_folder}, {count_origins}, {count_fraud}\n")
    
    mean_accuracy /= len(psp_folder_list)
    with open(result_file, "a") as f:
        f.write(f"Mean accuracy on test set : {mean_accuracy}\n")

