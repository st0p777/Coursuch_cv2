import os

import cv2
from matplotlib import pyplot

from settings import DATA_PATH, drawing, hands, drawing_styles
from lib import hands_obj


path = DATA_PATH

for letter_dir in os.listdir(path):
    for image_name in os.listdir(os.path.join(path, letter_dir)):
        image = cv2.imread(os.path.join(path, letter_dir, image_name))
        convert_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands_obj.process(convert_image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    convert_image,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        pyplot.figure()
        pyplot.imshow(convert_image)
        pyplot.show()

pyplot.close()
