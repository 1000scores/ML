import json


def set_description_json(df, y):
    """[Подразумевается что первые y.shape[0] элементов в df - это train часть (а за ней тестовая часть)]

    Args:
        df (DataFrame): [description]
        y (Series): [description]
    """
    target_to_okey = {'low': 0, 'medium': 0.5, 'high' : 1}
    description_words_weight = dict()
    description_words_count = dict()
    for index, target in y.items():
        row = df.iloc[index]
        for word in row.description:
            if len(word) < 3:
                continue
            if word in description_words_weight:
                description_words_weight[word] += target_to_okey[target]
                description_words_count[word] += 1
            else:
                description_words_weight[word] = target_to_okey[target]
                description_words_count[word] = 1
            
    for key, value in description_words_weight.items():
        description_words_weight[key] = value / (description_words_count[key])

    with open('description.json', 'w') as json_file:
        json.dump(description_words_weight, json_file)

def set_features_json(df, y):
    target_to_okey = {'low': 0, 'medium': 0.5, 'high' : 1}
    features_words_weight = dict()
    features_words_count = dict()
    for index, target in y.items():
        row = df.iloc[index]
        for word in row.features:
            word = str.lower(word)
            if word in features_words_weight:
                features_words_weight[word] += target_to_okey[target]
                features_words_count[word] += 1
            else:
                features_words_weight[word] = target_to_okey[target]
                features_words_count[word] = 1
            
    for key, value in features_words_weight.items():
        features_words_weight[key] = value / (features_words_count[key] * 100)

    print(features_words_weight['no fee'])
    with open('features.json', 'w') as json_file:
        json.dump(features_words_weight, json_file)
