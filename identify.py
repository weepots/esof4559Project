import face_recognition
import os
import PIL
import pandas as pd
from tabulate import tabulate
import cv2
import augmentor
from tqdm.auto import tqdm
from enhancer import Enhancer
import threading
import concurrent.futures

DATASET_LOCATION = './datasets'


def create_known_encodings(dataset_location):
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


def per_person_test(
        person,
        people_dictionary,
        blur_k_size,
        noise_sigma,
        cont_alpha,
        enhancer_um_k,
        enhancer_um_kernel,
        enhancer_um_gamma,
        enhancer_mb_kernel,
        enhancer_he_clip_limit,
        enhancer_he_tile_grid_size,
        augmentations,
        enhance_flag,
        save_flag):
    person_pictures_path = os.path.join(DATASET_LOCATION, person)
    person_pictures = os.listdir(person_pictures_path)
    failed_detections = 0
    failed_detection_enhanced = 0
    local_augmentations = augmentations
    for person_picture in person_pictures:
        test_image_path = os.path.join(
            DATASET_LOCATION, person, person_picture)
        augmented_image, changed_image_path, augmentations = augmentor.augment(
            input_image_path=test_image_path,
            image_name=person_picture,
            augmentations=local_augmentations,
            blur_k_size=blur_k_size,
            # need to change this
            cont_alpha=cont_alpha,
            noise_sigma=noise_sigma,
            # need to change this
        )

        enhancer = Enhancer()
        enhancer.um_k = enhancer_um_k
        enhancer.um_kernel = enhancer_um_kernel
        enhancer.um_gamma = enhancer_um_gamma
        enhancer.mb_kernel = enhancer_mb_kernel
        enhancer.he_clip_limit = enhancer_he_clip_limit
        enhancer.he_tile_grid_size = enhancer_he_tile_grid_size
        enhanced_image = enhancer.enhance(
            image=augmented_image, enhancements=local_augmentations)

        test_image_face_encodings = face_recognition.face_encodings(
            enhanced_image)
        test_image_face_encodings_unenhanced = face_recognition.face_encodings(
            augmented_image)
        match_results = matcher(test_image_face_encodings,
                                person, people_dictionary)
        match_results_unenhanced = matcher(test_image_face_encodings_unenhanced,
                                           person, people_dictionary)
        if save_flag:
            path_elements = changed_image_path.split('/')
            enhanced_image_folder_path = f"./{path_elements[1]}/{path_elements[2]}/enhanced/"
            if not os.path.exists(enhanced_image_folder_path):
                os.mkdir(enhanced_image_folder_path)
            enhanced_image_path = f"{enhanced_image_folder_path}/{path_elements[3]}"
            cv2.imwrite(enhanced_image_path, enhanced_image)
            cv2.imwrite(changed_image_path, augmented_image)

        if not match_results:
            failed_detections += 1
        if not match_results_unenhanced:
            failed_detection_enhanced += 1

    return failed_detections, failed_detection_enhanced, person


def tester2(
        blur_k_size,
        noise_sigma,
        cont_alpha,
        enhancer_um_k,
        enhancer_um_kernel,
        enhancer_um_gamma,
        enhancer_mb_kernel,
        enhancer_he_clip_limit,
        enhancer_he_tile_grid_size,
        augmentations,
        enhance_flag,
        save_flag
):
    people_dictionary = create_known_encodings(
        dataset_location=DATASET_LOCATION)
    # failed_detections_list = []
    failed_detections = 0
    failed_detections_unenhanced = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        future_person_failed_detections = {
            executor.submit(
                per_person_test,
                person,
                people_dictionary,
                blur_k_size,
                noise_sigma,
                cont_alpha,
                enhancer_um_k,
                enhancer_um_kernel,
                enhancer_um_gamma,
                enhancer_mb_kernel,
                enhancer_he_clip_limit,
                enhancer_he_tile_grid_size,
                augmentations,
                enhance_flag,
                save_flag
            ): person for person in people_dictionary}
        for future in concurrent.futures.as_completed(future_person_failed_detections):
            person_failed_detections, person_failed_detections_unenhanced, person = future.result()
            failed_detections += person_failed_detections
            failed_detections_unenhanced += person_failed_detections_unenhanced
            # failed_detections_list.append({person: person_failed_detections})
        executor.shutdown()
    # sequential
    # print(failed_detections_list)
    # print(len(failed_detections_list))

    # for person in people_dictionary:
    #     person_failed_detections, person = per_person_test(
    #         person,
    #         people_dictionary,
    #         blur_k_size,
    #         noise_sigma,
    #         enhancer_um_k,
    #         enhancer_um_kernel,
    #         enhancer_um_gamma,
    #         enhancer_mb_kernel,
    #         augmentations)
    #     failed_detections += person_failed_detections
    #     failed_detections_list.append({person: person_failed_detections})
    row = {
        'Blur kernel': f'{blur_k_size}',
        'Unsharp Mask K': f'{enhancer_um_k}',
        'Unsharp Mask Kernel': f'{enhancer_um_kernel}',
        'Unsharp Mask gamma': f'{enhancer_um_gamma}',
        'Noise Sigma': f'{noise_sigma}',
        'Median Blur Kernel': f'{enhancer_mb_kernel}',
        'Contrast Alpha': f'{cont_alpha}',
        'Histogram EQ Tile Size': f'{enhancer_he_tile_grid_size}',
        'Histogram EQ Clip Limit': f'{enhancer_he_clip_limit}',
        'Failed Detections': f'{failed_detections}',
        'Failed Detections (Unenhanced)': f'{failed_detections_unenhanced}'
    }
    # filtered_row = {key: value for key, value in row.items() if (
    #     value != '0' or key == 'Failed Detections')}
    return row


def mass_test_blur():

    blur_k_size_list = [5, 7, 9]
    enhancer_um_k_list = [3, 4, 5, 6, 7, 8]  # 4
    enhancer_um_kernel_list = [3]  # 3
    enhancer_um_gamma_list = [3, 4, 5, 6, 7, 8]  # 4
    augmentations = ['blur']
    save_flag = True
    enhance_flag = True

    test_results = pd.DataFrame(columns=['Blur kernel', 'Unsharp Mask K',
                                'Unsharp Mask Kernel', 'Unsharp Mask gamma', 'Failed Detections'])
    current_run = 0

    for blur_k_size in blur_k_size_list:
        for enhancer_um_k in enhancer_um_k_list:

            for enhancer_um_kernel in enhancer_um_kernel_list:
                # if len(test_results) > 2:
                #     prev_res_1 = test_results.loc[len(test_results)-1]
                #     prev_res_2 = test_results.loc[len(test_results)-2]
                #     if (
                #         prev_res_1['Blur kernel'] == prev_res_2['Blur kernel'] and
                #         prev_res_1['Unsharp Mask K'] == prev_res_2['Unsharp Mask K'] and
                #         prev_res_1['Unsharp Mask Kernel'] == prev_res_2['Unsharp Mask Kernel'] and
                #         prev_res_1['Unsharp Mask gamma'] != prev_res_2['Unsharp Mask gamma'] and
                #         prev_res_1['Failed Detections'] > prev_res_2['Failed Detections'] and

                #     ):

                #         print("break...")
                #         return
                with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                    future_row = {
                        executor.submit(
                            tester2,
                            blur_k_size=blur_k_size,
                            enhancer_um_k=enhancer_um_k,
                            enhancer_um_kernel=enhancer_um_kernel,
                            enhancer_um_gamma=enhancer_um_gamma,
                            noise_sigma=0.6,  # set default values
                            enhancer_mb_kernel=0.6,  # set default values
                            cont_alpha=1,  # set default values
                            enhancer_he_clip_limit=2.0,
                            enhancer_he_tile_grid_size=3,
                            augmentations=augmentations,
                            enhance_flag=enhance_flag,
                            save_flag=save_flag
                        ): enhancer_um_gamma for enhancer_um_gamma in enhancer_um_gamma_list}
                    for future in concurrent.futures.as_completed(future_row):
                        new_row = future.result()
                        test_results.loc[len(test_results)] = new_row
                        current_run += 1
                        print(current_run, new_row)

    test_results.to_csv('test_results_blur.csv')


def mass_test_noise():
    noise_sigma_list = [0.3, 0.4, 0.5, 0.6]
    enhancer_mb_kernel_list = [3]  # 3
    augmentations = ['noise']
    enhance_flag = True
    save_flag = True
    test_results = pd.DataFrame(
        columns=['Noise Sigma', 'Median Blur Kernel', 'Failed Detections'])
    counter = 0
    for noise_sigma in noise_sigma_list:
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            future_row = {
                executor.submit(
                    tester2,
                    blur_k_size=3,
                    cont_alpha=1.0,
                    enhancer_um_gamma=0,
                    enhancer_um_k=3,
                    enhancer_um_kernel=3,
                    noise_sigma=noise_sigma,
                    enhancer_mb_kernel=enhancer_mb_kernel,
                    enhancer_he_clip_limit=1,
                    enhancer_he_tile_grid_size=1,
                    augmentations=augmentations,
                    enhance_flag=enhance_flag,
                    save_flag=save_flag,
                ): enhancer_mb_kernel for enhancer_mb_kernel in enhancer_mb_kernel_list
            }
            for future in concurrent.futures.as_completed(future_row):
                new_row = future.result()
                test_results.loc[len(test_results)] = new_row
                counter += 1
                print(counter, new_row)
    test_results.to_csv('test_results_noise.csv')


def mass_test_contrast():
    cont_alpha_list = [0.1, 0.3, 0.5, 1, 1.5, 2]
    enhancer_he_clip_limit_list = [0.5, 0.7, 1]  # 3
    enhancer_he_tile_grid_size_list = [3, 5, 7, 9, 12]
    augmentations = ['contrast']
    enhance_flag = True
    save_flag = True
    test_results = pd.DataFrame(
        columns=['Contrast Alpha', 'Histogram EQ Tile Size', 'Histogram EQ Clip Limit', 'Failed Detections'])
    counter = 0
    for cont_alpha in cont_alpha_list:
        for enhancer_he_clip_limit in enhancer_he_clip_limit_list:
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                future_row = {
                    executor.submit(
                        tester2,
                        blur_k_size=3,
                        cont_alpha=cont_alpha,
                        noise_sigma=0.6,
                        enhancer_um_k=3,
                        enhancer_um_kernel=3,
                        enhancer_um_gamma=0,
                        enhancer_mb_kernel=3,
                        enhancer_he_clip_limit=enhancer_he_clip_limit,
                        enhancer_he_tile_grid_size=enhancer_he_tile_grid_size,
                        augmentations=augmentations,
                        enhance_flag=enhance_flag,
                        save_flag=save_flag
                    ): enhancer_he_tile_grid_size for enhancer_he_tile_grid_size in enhancer_he_tile_grid_size_list
                }
                for future in concurrent.futures.as_completed(future_row):
                    new_row = future.result()
                    test_results.loc[len(test_results)] = new_row
                    counter += 1
                    print(counter, new_row)

    test_results.to_csv('test_results_contrast.csv')


# test_results_filter = test_results.loc[test_results['Detected'] != True]
# percentage_detected = (len(test_results) -
#                        len(test_results_filter)) / len(test_results)
# print('Augmentations:')
# print(augmentations)
# print(f'Total test results: {len(test_results)}')
# print(f'Failed to detect: {len(test_results_filter)}')
# print(f'Percentage Detected: {percentage_detected * 100}')
# print(tabulate(test_results_filter, headers='keys', tablefmt='psql'))


def full_test():
    # blur
    blur_k_size_list = [3, 5, 7, 9]
    # unsharp Mask ideal hyperparameters
    enhancer_um_kernel = 3
    enhancer_um_k = 4
    enhancer_um_gamma = 4
    # noise
    noise_sigma_list = [0.3, 0.4, 0.5, 0.6]
    # median blur ideal hyperparameters
    enhancer_mb_kernel = 3
    # contrast
    cont_alpha_list = [0.1, 0.3, 0.5, 1, 1.5, 2]
    # CLAHE ideal hyperparameters
    enhancer_he_clip_limit = 1
    enhancer_he_tile_grid_size = 7
    # augmentations
    augmentations = ['blur', 'noise', 'contrast']
    # flags
    enhance_flag = True
    save_flag = False
    test_results = pd.DataFrame(columns=[
        'Blur kernel',
        'Unsharp Mask K',
        'Unsharp Mask Kernel',
        'Unsharp Mask gamma',
        'Noise Sigma',
        'Median Blur Kernel',
        'Contrast Alpha',
        'Histogram EQ Tile Size',
        'Histogram EQ Clip Limit',
        'Failed Detections',
        'Failed Detections (Unenhanced)'])
    current_run = 0
    for cont_alpha in cont_alpha_list:
        for noise_sigma in noise_sigma_list:
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                future_row = {
                    executor.submit(
                        tester2,
                        blur_k_size=blur_k_size,
                        enhancer_um_k=enhancer_um_k,
                        enhancer_um_kernel=enhancer_um_kernel,
                        enhancer_um_gamma=enhancer_um_gamma,
                        noise_sigma=noise_sigma,
                        enhancer_mb_kernel=enhancer_mb_kernel,
                        cont_alpha=cont_alpha,
                        enhancer_he_clip_limit=enhancer_he_clip_limit,
                        enhancer_he_tile_grid_size=enhancer_he_tile_grid_size,
                        augmentations=augmentations,
                        enhance_flag=enhance_flag,
                        save_flag=save_flag
                    ): blur_k_size for blur_k_size in blur_k_size_list}
                for future in concurrent.futures.as_completed(future_row):
                    new_row = future.result()
                    test_results.loc[len(test_results)] = new_row
                    current_run += 1
                    print(current_run, new_row)

    test_results.to_csv('test_results_total.csv')


if __name__ == '__main__':

    # testing for optimal unsharp mask hyperparameters
    # mass_test_blur()

    # testing for optimal median blur hyper parameters
    # mass_test_noise()

    # testing for optimal CLAHE hyper parameters
    # mass_test_contrast()

    # full test of combinations of augmentation parameters
    full_test()

    # individual hyperparameters test
    # print(tester2(
    #     noise_sigma=0.9,
    #     enhancer_mb_kernel=3,
    #     blur_k_size=3,
    #     cont_alpha=0.1,
    #     enhancer_um_gamma=0,
    #     enhancer_um_k=3,
    #     enhancer_um_kernel=3,
    #     enhancer_he_tile_grid_size=3,
    #     enhancer_he_clip_limit=2,
    #     augmentations=['contrast']
    # ))
