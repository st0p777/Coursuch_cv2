import os
import pickle
import cv2

import settings
from lib import get_prefix, hands_obj, process_image


path = settings.DATA_PATH


def image_analyse() -> tuple[list, list[str]]:
    temp_dataset = []
    letter_numbers = []
    for letter_dir in os.listdir(path):
        for img_path in os.listdir(os.path.join(path, letter_dir)):
            image = cv2.imread(os.path.join(path, letter_dir, img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands_obj.process(image_rgb)

            if result.multi_hand_landmarks:
                data = process_image(result)
                temp_dataset.append(data)
                letter_numbers.append(letter_dir)
            else:
                print(f"{letter_dir} - {img_path} not recognized")

    return temp_dataset, letter_numbers


dataset, numbers = image_analyse()

with open(f'datasets/dataset-{get_prefix(False)}', 'wb') as dataset_file:
    pickle.dump({'data': dataset, 'numbers': numbers}, dataset_file)
