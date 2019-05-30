import cv2
import numpy as np
import os

home = os.path.expanduser('~')
img_folder = home + '/Carla/scenario_runner_cz/image_npy'
img_list = os.listdir(img_folder)
for img in img_list:
    img_full_path = os.path.join(img_folder, img)
    if os.path.isfile(img_full_path):
        img_np = np.load(img_full_path )
        # print(img)
        img_filename = img.split('.')[0] + ".png"
        img_filename = os.path.join(img_folder, 'imgs', img_filename)
        print(img_filename)
        cv2.imwrite(img_filename, img_np)
        print(img + " converted to image")