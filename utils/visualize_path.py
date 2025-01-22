import matplotlib.pyplot as plt
import os
import json
import cv2

from utils.utils import draw_random_path

def visualize_path_from_json(image_name, json_path, hologram_mask_path):
    """
    Visualizes a random path on a hologram mask image and saves the result.

    Parameters:
    image_name (str): The name of the STR image to be visualized.
    json_path (str): The path to the JSON file containing the coordinates.
    hologram_mask_path (str): The file path to the hologram mask image.

    Returns:
    None
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

        filtered_data = {path: coords for path, coords in data.items() if image_name in path}

    list_coordinates = list(filtered_data.values())
    print("Number of random path in image: ", len(list_coordinates))
    image = draw_random_path(hologram_mask_path, list_coordinates)
    plt.imshow(image)
    plt.title(image_name)
    cv2.imwrite(f'visualization/{image_name}.jpg', image)

def visualize_all_frame(folder_path, json_path):
    """
    Visualizes all frames in a folder by drawing random paths on each frame based on coordinates from a JSON file.

    Parameters:
    folder_path (str): The path to the folder containing the frames.
    json_path (str): The path to the JSON file containing the coordinates.
    """
    folder_name = os.path.basename(folder_path)
    os.makedirs(f"visualization/{folder_name}", exist_ok=True)
    # Get all the paths in the json file that contains the folder name
    with open(json_path, 'r') as f:
        data = json.load(f)
        filtered_data = {path: coords for path, coords in data.items() if folder_name in path}

    list_coordinates = list(filtered_data.values())
    print("Number of random path in image: ", len(list_coordinates))

    list_img_name = os.listdir(folder_path)

    for name in list_img_name:
        frame_path = os.path.join(folder_path, name)
        image = draw_random_path(frame_path, list_coordinates)
        cv2.imwrite(f'visualization/{folder_name}/{name}', image)
        

if __name__ == "__main__":
    root_folder = "D:/MIDV"
    hologram_mask_path = root_folder + '/templates/hologram_masks/passport_hologram_mask_small.png'

    # file json contains coordinates of random paths
    json_path = "D:/MIDV/str_images_symmetric_copy/coordinates_from_origins.json" 
    frame_image_path = "D:/MIDV/extracted_passport/val/origins/passport/psp01_01_01"
    image_name = "psp01_01_01"
    # image_name = "psp09_02_01_str_0022_1024_595"

    # visualize_all_frame(frame_image_path, json_path)
    # image_path = "D:\\MIDV\\str_images_symmetric\\test\\origins\\psp01_02_01_str_0000_2151_1571.png"
    
    visualize_path_from_json(image_name, json_path, hologram_mask_path)
    # visualize_path_from_json("psp09_02_01_str_0022_1024_595", json_path, hologram_mask_path)
    plt.show()