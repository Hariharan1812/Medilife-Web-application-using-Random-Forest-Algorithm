import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier as Model
import pandas as pd


def algorithm(TrainCsv, Data):
    dataset = pd.read_csv(TrainCsv)

    a = []
    yt = []
    filename = f"{settings.MEDIA_ROOT}/dataset.csv"
    dataset = pd.read_csv(filename)
    prob = {'pain': 1, 'fever': 2, 'cold': 3, 'diabetes': 4, 'headache': 5}
    seve = {'low': 1, 'high': 3, 'medium': 2}
    xt = [[dosage, prob[problem], seve[severity]]]
    for i in range(1, 10):
        x = dataset.iloc[:, 2:]
        y = dataset.iloc[:, 1:2]
        labelencoder = LabelEncoder()
        ytrain = labelencoder.fit_transform(y)
        std = StandardScaler()
        xtrain = std.fit_transform(x)

        classifier = RandomForestClassifier()
        classifier.fit(x, ytrain)

        ypred = classifier.predict(xt)
        ypredicted = labelencoder.inverse_transform(ypred)
        print(ypredicted)
        a.append(ypredicted)

    for i in a:
        if i not in yt:
            yt.append(i[0])

    return yt
    data_x = dataset.iloc[:, :-1]
    data_y = dataset.iloc[:, -1]
    rendom_number = random.randrange(0, len(dataset))
    random_data = dataset.iloc[rendom_number, :-1]
    sample_x = list(data_x.iloc[0])
    sample_y = data_y.iloc[0]

    string_columns = []
    for i in sample_x:
        if type(i) == str:
            string_columns.append(sample_x.index(i))
    yLabel = False
    if type(sample_y) == str:
        yLabelencoder = LabelEncoder()
        yLabel = True
        data_y = yLabelencoder.fit_transform(data_y)

    LabelEncoders = []

    for i in string_columns:
        newLabelEncoder = LabelEncoder()
        data_x.iloc[:, i] = newLabelEncoder.fit_transform(data_x.iloc[:, i])
        LabelEncoders.append(newLabelEncoder)
    model = Model()
    model.fit(data_x, data_y)
    l = 0
    # a = LabelEncoders[0]
    new_data = Data
    for i in string_columns:
        z = LabelEncoders[l]
        new_data.iloc[:, i] = newLabelEncoder.fit_transform(new_data.iloc[:, i])
        l += 1

    predicted = model.predict(new_data)
    if yLabel is True:
        predicted = yLabelencoder.inverse_transform(predicted)
    new_data['output'] = predicted
    return new_data
