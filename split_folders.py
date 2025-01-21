import numpy as np
import cv2
import json
import os
from os.path import join as pjoin, basename, dirname
import random
import shutil
import matplotlib.pyplot as plt

'''
Create a list of folders for the train, val and test sets
64% train, 16% validation and 20% test
96 videos for train, 24 videos for validation and 30 for test
'''

random.seed(20)
# Select index for test set randomly
test_index_list = [random.choice(range(1, 6)) for _ in range(10)]
val_index_list = [random.choice([x for x in range(1,6) if x != test_index_list[i]]) for i in range(8)]

print(test_index_list)
print(val_index_list)

# Define the root folder of the dataset and the paths to the images and markup folders
root_folder = "D:/MIDV"
markup_p = pjoin(root_folder, "markup")
images_p = pjoin(root_folder, "images")

# Define the paths to the original and fraud images
origin_path = pjoin(images_p, 'origins', 'passport')

fraud_types = ['copy_without_holo', 'photo_holo_copy', 'pseudo_holo_copy']
fraud_paths = [pjoin(images_p, 'fraud', x, 'passport') for x in fraud_types]


# Get the list of folders in the original images folder
folders = os.listdir(origin_path)

# Split into train and test sets
train_set = []
val_set = []
test_set = []

# Define the list of psp types
psp_types = ['psp' + i for i in [f"{num:02}" for num in range(1,11)]]

# Create test set folders
for i, psp_type in enumerate(psp_types):
    test_pre = '_'.join([psp_type, f"{test_index_list[i]:02}"])
    test_set += [folder for folder in folders if folder.startswith(test_pre)]

# Create val set folders
for i, psp_type in enumerate(psp_types[:8]):
    val_pre = '_'.join([psp_type, f"{val_index_list[i]:02}"])
    val_set += [folder for folder in folders if folder.startswith(val_pre)]

# Train set folder
train_set = [folder for folder in folders if (folder not in test_set) and (folder not in val_set)]

os.makedirs("list_folders", exist_ok=True)

# Save the test set to test.txt
with open("list_folders/origins_test.lst", "w") as test_file:
    test_lst = ['/'.join(['passport', folder]) for folder in test_set]
    test_file.write("\n".join(test_lst))
    # test_file.write("\n".join(test_set))

    # Save the test set to test.txt
with open("list_folders/origins_val.lst", "w") as val_file:
    val_lst = ['/'.join(['passport', folder]) for folder in val_set]
    val_file.write("\n".join(val_lst))
    # val_file.write("\n".join(val_set))

# Save the train set to train.txt
with open("list_folders/origins_train.lst", "w") as train_file:
    train_lst = ['/'.join(['passport', folder]) for folder in train_set]
    train_file.write("\n".join(train_lst))
    # train_file.write("\n".join(train_set))
