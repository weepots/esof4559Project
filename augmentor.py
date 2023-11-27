import cv2
import os
import numpy as np


def blur(image, k_size):
    kernel = (k_size, k_size)
    changed_image = cv2.blur(image, kernel)
    return changed_image


def contrast(image, alpha):
    changed_image = cv2.convertScaleAbs(image, alpha=alpha)
    return changed_image


def noise(image, sigma):
    mean = 0
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    changed_image = cv2.add(image, noise)
    return changed_image


# def exposure(image, brightness_factor):
#     changed_image = np.clip(
#         (image + brightness_factor), 0, 255).astype(np.uint8)
#     return changed_image


def augment(
        input_image_path,
        image_name,
        augmentations,
        blur_k_size,
        cont_alpha,
        noise_sigma,
        exp_brightness_factor=1):
    changed_image = cv2.imread(input_image_path)
    # changed_image = cv2.cvtColor(changed_image, cv2.COLOR_RGB2GRAY)
    # changed_image = cv2.cvtColor(changed_image, cv2.COLOR_GRAY2RGB)
    augmentations_results = []
    runs_directory = './runs/run'
    for change in augmentations:
        if change == 'blur':
            changed_image = blur(
                image=changed_image, k_size=blur_k_size)
            runs_directory += '_blur_' + str(blur_k_size)
            augmentations_results.append({"Blur": f"ksize: {blur_k_size}"})
        if change == 'contrast':
            changed_image = contrast(
                image=changed_image, alpha=cont_alpha)
            runs_directory += '_con_' + str(cont_alpha)
            augmentations_results.append({"Contrast": f"alpha: {cont_alpha}"})
        if change == 'noise':
            changed_image = noise(
                image=changed_image, sigma=noise_sigma)
            runs_directory += '_noise_' + str(noise_sigma)
            augmentations_results.append({"noise": f"sigma: {noise_sigma}"})
        # if change == 'exposure':
        #     changed_image = exposure(
        #         image=changed_image, brightness_factor=exp_brightness_factor)
        #     runs_directory += '_exp_' + str(exp_brightness_factor)
        #     augmentations_results.append(
        #         {"Exposure": f"Brightness Factor: {exp_brightness_factor}"})
    if not os.path.exists(runs_directory):
        os.mkdir(runs_directory)
    changed_image_path = os.path.join(runs_directory, image_name)
    return changed_image, changed_image_path, augmentations_results


# def augment(changes, input_image_path, image_name):
#     test_image_to_change = cv2.imread(input_image_path)
#     # test_image_to_change = cv2.cvtColor(
#     #     test_image_to_change, cv2.COLOR_RGB2GRAY)
#     ksize = (3, 3)
#     changed_image = test_image_to_change
#     alpha = 0.1  # Contrast control
#     # brightness_factor = -50
#     sigma = 0.5
#     runs_directory = './runs/run'
#     augmentations = []
#     for change in changes:
#         if change == 'blur':
#             changed_image = cv2.blur(changed_image, ksize)
#             runs_directory += '_blur_' + str(ksize[0])
#             augmentations.append({"Blur": f"ksize: {ksize}"})
#         if change == 'contrast':
#             changed_image = cv2.convertScaleAbs(
#                 changed_image, alpha=alpha,)
#             runs_directory += '_con_' + str(alpha)
#             augmentations.append({"Contrast": f"alpha: {alpha}"})
#         # if change == 'exposure':
#         #     changed_image = np.clip(
#         #         (changed_image + 10), 0, 255).astype(np.uint8)
#         #     runs_directory += '_exp_' + str(brightness_factor)
#         #     augmentations.append(
#         #         {"Exposure": f"Brightness Factor: {brightness_factor}"})
#         if change == 'noise':
#             mean = 0
#             noise = np.random.normal(
#                 mean, sigma, changed_image.shape).astype(np.uint8)
#             changed_image = cv2.add(changed_image, noise)
#             runs_directory += '_noise_' + str(sigma)
#             augmentations.append(
#                 {"noise": f"sigma: {sigma}"})
#         # if change == 'cut off half image':
#     if not os.path.exists(runs_directory):
#         os.mkdir(runs_directory)

#     changed_image_path = os.path.join(runs_directory, image_name)
#     return changed_image, changed_image_path, augmentations

# def
