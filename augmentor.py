import cv2
import face_recognition
import os


def augment(changes, input_image_path, image_name, runs_directory):
    test_image_to_change = cv2.imread(input_image_path)
    ksize = (5, 5)
    changed_image = test_image_to_change
    alpha = 0.5  # Contrast control
    for change in changes:
        if change == 'blur':
            changed_image = cv2.blur(changed_image, ksize)
        if change == 'contrast':
            changed_image = cv2.convertScaleAbs(
                changed_image, alpha=alpha)
        # if change == 'noise':
        # if change == 'cut off half image':
    changed_image_path = os.path.join(runs_directory, image_name)
    return changed_image, changed_image_path
