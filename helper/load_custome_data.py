import cv2
import os
import numpy as np

def load_images_from_folder(folder, width, height, shuffle=True):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (width, height))
            images.append(img)
    images = np.array(images)
    if shuffle:
    	np.random.shuffle(images)
    return images