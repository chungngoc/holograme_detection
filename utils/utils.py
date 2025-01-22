import numpy as np
import os
import cv2
import json
import random

def select_random_pixel(image, with_holo = True, margin=100):
  '''Select random pixel far enough from the edges'''
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
  return tuple(random.choice(valid_pixels))

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

def create_random_path(image_path, with_holo =True, margin = 100, path_length=224):
  """
    From a pixel, choose 1 pixel randomly from 8 pixels around it.
  """
  # Load the grayscale image
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if image is None:
    raise ValueError("Error: Could not load the image.")

  start_pixel = select_random_pixel(image, with_holo, margin)
  path = [start_pixel]

  while len(path) < path_length:
    current_pixel = path[-1]
    neighbors = get_neighbors(current_pixel, image.shape, margin)

    # Choose the next pixel randomly from neighbors
    next_pixel = random.choice(neighbors)
    path.append(next_pixel)

  return path

def draw_random_path(image_path, list_coordinates, path_color=(255, 0, 0), box_color=(0, 255, 0)):
  image = cv2.imread(image_path)

  for coordinates in list_coordinates:
    # Draw the path on the image
    for pixel in coordinates:
      image[pixel[0], pixel[1]] = path_color

    # Find the bounding box around the path
    rows, cols = zip(*coordinates)
    top_left = (min(cols)- 5, min(rows)- 5)  # (x_min, y_min)
    bottom_right = (max(cols)+5, max(rows)+5)  # (x_max, y_max)

    cv2.rectangle(image, top_left, bottom_right, box_color, 2)
  return image

def create_symmetric_stack(small_image, target_width=224):
  """
  Stacks the small image symmetrically to create an image of the specified width.

  Parameters:
    small_image: Input image as a numpy array of shape (H, W, C).
    target_width: Desired width of the output image.

  Returns:
    stacked_image: Symmetrically stacked image of shape (H, target_width, C).
  """
  small_width = small_image.shape[1]

  # Calculate how many full repeats of the small image are needed
  num_full_repeats = target_width // small_width
  leftover_width = target_width % small_width

  # Initialize an empty list for stacking
  images = []

  # Add full repeats with alternating symmetry
  for i in range(num_full_repeats):
    if i % 2 == 0:
      images.append(small_image)  # Original image
    else:
      images.append(np.flip(small_image, axis=1))  # Horizontally flipped image

  # Handle the leftover width
  if leftover_width > 0:
    # Create a mirrored or original slice for the leftover part
    if num_full_repeats % 2 == 0:
      leftover_image = small_image[:, :leftover_width, :]
    else:
      leftover_image = np.flip(small_image, axis=1)[:, :leftover_width, :]
    images.append(leftover_image)

  # Horizontally stack all the parts
  stacked_image = np.hstack(images)
  return stacked_image

def stack_STR(STR, target_width=224):
  left_over = STR[:, :target_width % STR.shape[1]]
  STR = np.hstack([STR] * (target_width // STR.shape[1]))
  return np.hstack((STR, left_over))

def save_random_paths(file_path, image_path, coordinates):
  """
  Saves image paths and coordinates of pixels to a JSON file, appending new entries.

  Parameters:
      file_path: Path to the JSON file.
      image_path: The image path (string).
      coordinates: List of tuples representing coordinates of random pixels[(x1, y1), (x2, y2), ...].
  """
  # Initialize data
  data = {}
  # Convert NumPy data types to Python types for JSON compatibility
  coordinates = [[int(x), int(y)] for x, y in coordinates]

  # Load existing data if the file exists
  if os.path.exists(file_path):
    with open(file_path, 'r') as f:
      data = json.load(f)

  # Append new data
  data[image_path] = coordinates

  # Save updated data back to the JSON file
  with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
