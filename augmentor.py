import cv2
import os
import numpy as np


def augment(changes, input_image_path, image_name):
    test_image_to_change = cv2.imread(input_image_path)
    ksize = (5, 5)
    changed_image = test_image_to_change
    alpha = 0.5  # Contrast control
    brightness_factor = -127
    runs_directory = './runs/run'
    augmentations = []
    for change in changes:
        if change == 'blur':
            changed_image = cv2.blur(changed_image, ksize)
            runs_directory += '_blur_' + str(ksize[0])
            augmentations.append({"Blur": f"ksize: {ksize}"})
        if change == 'contrast':
            changed_image = cv2.convertScaleAbs(
                changed_image, alpha=alpha)
            runs_directory += '_con_' + str(alpha)
            augmentations.append({"Contrast": f"alpha: {alpha}"})
        if change == 'exposure':
            changed_image = np.clip(
                changed_image + brightness_factor, 0, 255).astype(np.uint8)
            runs_directory += '_exp_' + str(brightness_factor)
            augmentations.append(
                {"Exposure": f"Brightness Factor: {brightness_factor}"})
        if change == 'noise':
            row, col, ch = changed_image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            changed_image = changed_image + gauss
        # if change == 'cut off half image':
    if not os.path.exists(runs_directory):
        os.mkdir(runs_directory)

    changed_image_path = os.path.join(runs_directory, image_name)
    return changed_image, changed_image_path, augmentations
