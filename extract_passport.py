import cv2
import numpy as np
import json
import os

def extract_passport_from_frame(frame_path, markup_path, dim, output_folder):
  '''
  frame_path : path of folder containts frames
  markup_path : path of folder containts files json of markup
  dim : Image size of hologram mask
  output_folder : path folder to save images of passport
  '''
  # Check if the output folder exists
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Load the frame image
  frame = cv2.imread(frame_path)
  if frame is None:
    print(f"Error: Could not load frame {frame_path}")
    return

  # Load JSON data
  with open(markup_path, 'r') as f:
    data = json.load(f)

  # Get the four corner points
  type_passport = data["document"]["document_type"] + ":main"
  template_quad = data["document"]["templates"][type_passport]["template_quad"]
  pts_src = np.array(template_quad, dtype=np.float32)

  # Define the width and height for the transformed passport image
  width, height = dim[0], dim[1]

  # Define destination points for perspective transformation
  pts_dst = np.array([
      [0, 0],
      [width - 1, 0],
      [width - 1, height - 1],
      [0, height - 1]
  ], dtype=np.float32)

  # Compute the perspective transform matrix
  H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
  # matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

  # Get the passport image
  passport_image = cv2.warpPerspective(frame, H, (width, height))
  # passport_image = cv2.resize(passport_image, dim, interpolation = cv2.INTER_AREA)

  # Save the extracted passport image
  frame_name = os.path.splitext(os.path.basename(frame_path))[0]
  passport_image_path = os.path.join(output_folder, f"{frame_name}.jpg")
  cv2.imwrite(passport_image_path, passport_image)


if __name__ == '__main__':
    root_folder = "D:/MIDV"
    # Define the path to the list of folders, after split to train, val, test
    # After run split_folders.py
    path_to_split_folders = "C:/Users/ADMIN/python/holograme_detection/split_folders"

    # Path of hologram mask
    hologram_mask_path = os.path.join(root_folder, 'templates/hologram_masks/passport_hologram_mask_small.png')
    hologram_mask = cv2.imread(hologram_mask_path, cv2.IMREAD_GRAYSCALE)
    dim = hologram_mask.shape[::-1]

    # Define the type of folders
    type_folders = ['fraud/copy_without_holo', 'fraud/photo_holo_copy', 'fraud/pseudo_holo_copy'] 
    
    ##############################################################################
    for type_folder in type_folders:
        # Loop through the split folders
        for file_name in os.listdir(path_to_split_folders):
            type_split = file_name.split('_')[1][:-4]  # train, val, test
            
            # Read the list of folders from the file
            with open(os.path.join(path_to_split_folders, file_name), 'r') as f:
                folders = [line.strip() for line in f.readlines()]
            
            # Loop through each folder in the list
            for passport_folder in folders:
                psp_folder = passport_folder.split('/')[1]
                if psp_folder in os.listdir(os.path.join(root_folder,'images', type_folder, 'passport')):
                  list_images_path =  os.path.join(root_folder,'images', type_folder, passport_folder, 'list.lst')
                  
                  # Get all image name
                  with open(list_images_path, 'r') as file:
                      file_list = [line.strip() for line in file if line.strip()]

                      # Loop through frames
                      for image_name in file_list:
                          frame_path = os.path.join(root_folder,'images', type_folder, passport_folder, image_name)
                          markup_path = os.path.join(root_folder, 'markup', type_folder, passport_folder, image_name + '.json')

                          output_folder = os.path.join(root_folder, 'extracted_passport', type_split, type_folder, passport_folder)
                          # Extract passport from frame
                          extract_passport_from_frame(frame_path, markup_path, dim, output_folder)