import mediapipe


DATA_PATH = 'learn_data/alphabet'

ALPHABET = {
    1: 'а',
    2: 'б',
    3: 'в',
    4: 'г',
    5: 'д',
    6: 'е',
    7: 'ё',
    8: 'ж',
    9: 'з',
    10: 'и',
    11: 'й',
    12: 'к',
    13: 'л',
    14: 'м',
    15: 'н',
    16: 'о',
    17: 'п',
    18: 'р',
    19: 'с',
    20: 'т',
    21: 'у',
    22: 'ф',
    23: 'х',
    24: 'ц',
    25: 'ч',
    26: 'ш',
    27: 'щ',
    28: 'ь',
    29: 'ъ',
    30: 'ы',
    31: 'э',
    32: 'ю',
    33: 'я',
}

CLASSES_COUNT = 27

input_shape = (  # Размер тензора на основе изображения для входных данных в нейронную сеть
    42,
    42,
    1,  # количество каналов для цвета
)

solution = mediapipe.solutions
hands = solution.hands
drawing = solution.drawing_utils
drawing_styles = solution.drawing_styles
