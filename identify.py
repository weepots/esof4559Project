import face_recognition
import os
import PIL
import pandas as pd
from tabulate import tabulate
import cv2
import augmentor
import tqdm


DATASET_LOCATION = './datasets'
# image_adjustments = ['blur', 'exposure']
image_adjustments = ['noise']
# image_adjustments = []


def create_known_encodings(dataset_location):
    dataset_location = './datasets'
    people_list = [f for f in os.listdir(
        dataset_location) if not f.startswith('.')]
    people_dictionary = {}
    for person in people_list:
        people_dictionary[person] = None
    for person in people_dictionary:
        # Use first image of person as reference image
        reference_image_name = person + '_0001.jpg'
        reference_image_path = os.path.join(
            dataset_location, person, reference_image_name)

        # Show image
        # reference_image = PIL.Image.open(reference_image_path)
        # reference_image.show()

        # load image, generate facial encodings and store as a variable
        reference_image = face_recognition.load_image_file(
            reference_image_path)
        reference_image_face_encoding = face_recognition.face_encodings(
            reference_image)
        people_dictionary[person] = reference_image_face_encoding
    return people_dictionary


def matcher(test_image_face_encodings, person, people_dictionary):
    for test_image_face_encoding in test_image_face_encodings:
        matches = face_recognition.compare_faces(
            people_dictionary[person], test_image_face_encoding)
        if matches == None:
            return False
        if True in matches:
            return True
        else:
            return False


people_dictionary = create_known_encodings(dataset_location=DATASET_LOCATION)
test_results = pd.DataFrame(columns=['Image name', 'Detected'])
augmentations = []
for person in tqdm.tqdm(people_dictionary):
    person_pictures_path = os.path.join(DATASET_LOCATION, person)
    person_pictures = os.listdir(person_pictures_path)
    for person_picture in person_pictures:
        test_image_path = os.path.join(
            DATASET_LOCATION, person, person_picture)
        test_image, changed_image_path, augmentations = augmentor.augment(
            changes=image_adjustments, input_image_path=test_image_path, image_name=person_picture)
        test_image_face_encodings = face_recognition.face_encodings(test_image)
        match_results = matcher(test_image_face_encodings,
                                person, people_dictionary)
        cv2.imwrite(changed_image_path, test_image)
        # if not match_results:
        #     cv2.imwrite(changed_image_path, test_image)

        new_row = {'Image name': person_picture, 'Detected': match_results}
        test_results.loc[len(test_results)] = new_row


test_results_filter = test_results.loc[test_results['Detected'] != True]
percentage_detected = (len(test_results) -
                       len(test_results_filter)) / len(test_results)
print('Augmentations:')
print(augmentations)
print(f'Total test results: {len(test_results)}')
print(f'Failed to detect: {len(test_results_filter)}')
print(f'Percentage Detected: {percentage_detected * 100}')
print(tabulate(test_results_filter, headers='keys', tablefmt='psql'))
