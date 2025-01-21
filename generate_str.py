from utils import create_random_path, create_symmetric_stack, stack_STR, save_random_paths
import os
import cv2
import numpy as np

def get_STR(random_path, mask_image_shape, list_image_path):
    '''
    Use Vectorized operations

    random_path : list of (x, y) pixel coordinates in mask_image
    mask_image_shape : shape of the mask image
    list_image_path : list of paths to all frames
    return: numpy array of shape (len(random_path), len(list_image_path), 3)
    '''
    # STR : Spatio-Temporal Representation
    # Preallocate array for STR
    STR = np.empty((len(random_path), len(list_image_path), 3), dtype=np.uint8)

    random_path_array = np.array(random_path)
    rows, cols = random_path_array[:, 0], random_path_array[:, 1]
    
    # Iterate through all frame paths
    for i, path in enumerate(list_image_path):
        frame_image = cv2.imread(path, cv2.IMREAD_COLOR)  # Read directly in color
        if frame_image.shape[:2] != mask_image_shape:
            frame_image = cv2.resize(frame_image, mask_image_shape, interpolation=cv2.INTER_AREA)

        STR[:, i, :] = frame_image[rows, cols, :]
        
    # Change to RGB color
    # STR = STR[..., ::-1]

    return STR

def create_str(hologram_mask_path,
               frame_dir,
               save_folder,
               json_path,
               with_holo=None,
               num_str=50,
               symmetric=False
    ):
    '''
    Create STR and duplicate to create a 224x224x3 image
    Parameters:
        hologram_mask_path : path to the hologram mask
        frame_dir : path to the folder containing frames of the video
        save_folder : path to the folder to save the STR
        json_path : path to the json file to save the coordinates of pixels
        with_holo : True if the STR contains hologram, False if not, None if it's for evaluation
        num_str : number of STR to create
        symmetric : True if the STR is symmetric, False if just stack the STR
    '''
    list_image_path = [os.path.join(frame_dir, x) for x in os.listdir(frame_dir) if x.endswith(('.jpg', '.png'))]
    psp_folder = os.path.basename(frame_dir)


    mask_image = cv2.imread(hologram_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_image_shape = mask_image.shape

    k = 0
    while k < num_str:
        random_path = create_random_path(hologram_mask_path,with_holo, path_length=224)

        # Check if the path contains enough hologram pixels
        hologram_pixel_count = np.sum(mask_image[tuple(zip(*random_path))] > 0)
        # Validate path based on hologram pixel count
        if (with_holo and hologram_pixel_count < 20) or (not with_holo and hologram_pixel_count > 0):
            continue
        
        # Create STR
        STR = get_STR(random_path, mask_image_shape, list_image_path)
        
        if symmetric:
            STR = create_symmetric_stack(STR, target_width=224)
        else:
            STR = stack_STR(STR, target_width=224)

        # Save STR to folder  
        initial_point = random_path[0]
        initial_name = '_'.join(['str', f"{k:04}", str(initial_point[0]), str(initial_point[1])])

        str_name = '_'.join([psp_folder, initial_name])
        str_image_path = os.path.join(save_folder, f"{str_name}.jpg")
        cv2.imwrite(str_image_path, STR)

        # Save the path to a text file
        save_random_paths(json_path, str_image_path, random_path)

        k+=1


if __name__ == '__main__':
    root_folder = "D:/MIDV" # Need to change to adapt
    str_folder_name = "str_images_symmetric"
    fraud_folder = ["fraud/copy_without_holo", "fraud/photo_holo_copy", "fraud/pseudo_holo_copy"]
    type_folder = "origins"    # Create STR from origins videos
    # We can create STR from fraud videos by changing type_folder = "fraud/copy_without_holo" for example

    with_holo = True # Change to create STR in zone hologram or without hologram

    hologram_mask_path = os.path.join(root_folder, 'templates/hologram_masks/passport_hologram_mask_small.png')

    for type in ['test', 'val', 'train']:
        # Path to directory contains all folders of passeport images, after run extract_passport.py
        path_folders = os.path.join(root_folder, f"extracted_passport/{type}/{type_folder}/passport")

        if with_holo: 
            num_str = 50
        else:
            num_str = 30
            type_folder = "fraud"

        save_folder = os.path.join(root_folder, str_folder_name, type, type_folder)
        # Check if the output folder exists
        os.makedirs(save_folder, exist_ok=True)

        json_name = f"coordinates_{type_folder}.json" # Need to define
        json_name = json_name.replace('/', '_') # In case type_folder = "fraud/copy_without_holo" for example.
        json_path = os.path.join(root_folder, str_folder_name, type, json_name)

        for folder in os.listdir(path_folders):
            frame_dir = f"{path_folders}/{folder}" # Ex : "extracted_passport/test/origins/passport/psp01_01_01"
            # Create STR
            create_str(hologram_mask_path, frame_dir, save_folder, json_path, with_holo=with_holo, num_str = num_str, symmetric=True)
