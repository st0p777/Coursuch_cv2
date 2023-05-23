import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import asarray

from lib import get_prefix
# from convolutional_models import model_1_level, model_2_level


data_dict = pickle.load(open(f'datasets/dataset-{get_prefix(False)}', 'rb'))
labels = asarray(data_dict['numbers'])

x_train, x_test, y_train, y_test = train_test_split(
    asarray(data_dict['data']),
    labels,
    test_size=0.25,
    shuffle=True,
    stratify=labels,
)

models = [
    RandomForestClassifier,
    # model_1_level,
    # model_2_level,
]

dict_with_models_result = {}

for item in range(len(models)):
    model = models[item]()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    dict_with_models_result.update(
        {
            f'item_{item}': {
                'model': model,
                'score': score,
            }
        }
    )

best_score = max(dict_with_models_result, key=dict_with_models_result.get('score'))
best_model = dict_with_models_result[best_score]

print('{}% of samples were classified correctly from best model!'.format(best_model['score'] * 100))

with open(f'models/model-{get_prefix(True)}', 'wb') as model_file:
    pickle.dump({'model': best_model['model']}, model_file)
