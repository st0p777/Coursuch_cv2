import cv2
import pickle
import numpy as np

from lib import get_prefix, hands_obj, process_image
from settings import ALPHABET


model = pickle.load(open(f'models/model-{get_prefix(False)}', 'rb'))['model']
cap = cv2.VideoCapture(2)


while True:
    ret, frame = cap.read()
    result = hands_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        data = process_image(result)
        prediction = model.predict([np.asarray(data)])
        print(ALPHABET[int(prediction[0])])
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
