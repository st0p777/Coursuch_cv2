import mediapipe

hands_obj = mediapipe.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def get_prefix(update) -> int:
    number = int(open("checkpoint", mode='r').read())
    if update:
        open("checkpoint", mode='w').write(f"{number + 1}")
    return number


def process_image(landmark) -> list:
    data, x, y = [], [], []
    for hand_landmarks in landmark.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x.append(hand_landmarks.landmark[i].x)
            y.append(hand_landmarks.landmark[i].y)

        for i in range(len(hand_landmarks.landmark)):
            data.append(hand_landmarks.landmark[i].x - min(x))
            data.append(hand_landmarks.landmark[i].y - min(y))
    return data
