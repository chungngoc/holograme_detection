import os
import cv2
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_symmetric_stack, stack_STR, save_random_paths

def select_random_pixels(image, with_holo = True, margin=100, number_pixels =1):
  """
    Select multiple random pixels far enough from the edges.

    Parameters:
        image: Grayscale image (2D array).
        number_paths: Number of starting pixels to select.
        with_holo: Whether to select pixels in or outside the hologram zone.
        margin: Margin from the image edges.

    Returns:
        start_pixels: List of tuples representing the starting pixels.
    """
  
  height, width = image.shape

  # Define the valid area
  valid_area = image[margin:height - margin, margin:width - margin]
  if with_holo:
    # Select pixels in zone hologram
    valid_pixels = np.argwhere(valid_area > 0)
  elif with_holo is None:
    # Select any pixel
    valid_pixels = np.argwhere(np.ones_like(valid_area, dtype=bool))
  else:
    # Select pixels not in the hologram area
    valid_pixels = np.argwhere(valid_area == 0)

  # Adjust coordinates back to the original image space
  valid_pixels += margin
  # Randomly choose starting pixels
  start_pixels = random.sample(list(valid_pixels), number_pixels)
  return start_pixels

def get_neighbors(pixel, image_shape, margin):
  # Define 8 possible directions (row, col) offsets
  directions = [(-1, -1), (-1, 0), (-1, 1),(0, -1),(0, 1),(1, -1), (1, 0), (1, 1)]

  height, width = image_shape
  neighbors = []

  for dr, dc in directions:
    new_r, new_c = pixel[0] + dr, pixel[1] + dc
    if (margin <= new_r < height - margin) and (margin <= new_c < width - margin):
      neighbors.append((new_r, new_c))

  return neighbors

def create_random_path(image_path, with_holo =True, margin = 100, path_length=224, number_paths=1):
  """
    Generate multiple random paths in the image.

    Parameters:
        image_path: Path to the grayscale image.
        number_paths: Number of paths to generate.
        with_holo: Whether to select pixels in or outside the hologram zone.
        margin: Margin from the image edges.
        path_length: Length of each path.

    Returns:
        paths: List of paths, where each path is a list of pixel coordinates.
    """
  # Load the grayscale image
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if image is None:
    raise ValueError("Error: Could not load the image.")

  # Select starting pixels
  start_pixels = select_random_pixels(image, with_holo, margin, number_pixels=number_paths)

  paths = []
  for start_pixel in start_pixels:
      path = [start_pixel]

      while len(path) < path_length:
          current_pixel = path[-1]
          neighbors = get_neighbors(current_pixel, image.shape, margin)

          # Choose the next pixel randomly from neighbors
          next_pixel = random.choice(neighbors)
          path.append(next_pixel)

      paths.append(path)
  return paths

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

    # Generate all random paths at once
    random_paths = create_random_path(hologram_mask_path, with_holo, path_length=224, number_paths=num_str)

    # Process each random path
    for k, random_path in enumerate(random_paths):
        # Create STR
        STR = get_STR(random_path, mask_image_shape, list_image_path)

        # Apply symmetric or stacking logic
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

        # Save the path to a JSON file
        save_random_paths(json_path, str_image_path, random_path)

if __name__ == '__main__':
    root_folder = "D:/MIDV"

    hologram_mask_path = os.path.join(root_folder, 'templates/hologram_masks/passport_hologram_mask_small.png')
    # path_folders : path to directory contains passport folder of test set.
    path_folders = "D:/MIDV/extracted_passport/test/origins/passport"
    # path_folders = "D:/MIDV/extracted_passport/test/fraud/copy_without_holo/passport"

    # Define where to save STR
    save_folder_dir = 'str_images_evaluate/evaluate_origins'
    # save_folder_dir = 'str_images_evaluate/evaluate_copy_without_holo'

    psp_folders = os.listdir(path_folders)

    for folder in psp_folders:
        save_folder = os.path.join(root_folder, save_folder_dir, folder)
        json_path = os.path.join(save_folder, 'coordinates.json')

        # Check if the output folder exists
        os.makedirs(save_folder, exist_ok=True)
        
        frame_dir = f"{path_folders}/{folder}"
        # Create 1000 random paths 
        create_str(hologram_mask_path, frame_dir, save_folder, json_path, num_str = 1000, symmetric=True)
