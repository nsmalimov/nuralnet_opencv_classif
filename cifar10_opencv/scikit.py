import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def read_pickle_file(filename):
    f = open(filename, "r")
    data = pickle.load(f)
    f.close()

    return data


def pickle_write(data, filename):
    f = open(filename, "w")

    pickle.dump(data, f)

    f.close()


def train_test():
    train_images = read_pickle_file("images_train.pkl")

    all_classes = []

    for i in train_images:
        all_classes.append(train_images[i][1])

    print(len(set(all_classes)))

    train_images_new = []

    for i in train_images:
        train_images_new.append([train_images[i][0], train_images[i][1]])

    train_images = train_images_new

    np.random.shuffle(train_images)

    rf = RandomForestClassifier(n_jobs=4)

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    num = 0
    for i in train_images:
        if (num >= 40000):
            X_test.append(i[0])
            Y_test.append(i[1])
        else:
            X_train.append(i[0])
            Y_train.append(i[1])

        num += 1

    print(len(X_test))
    print(len(Y_test))

    print(Y_test)

    print("start fit")
    rf.fit(X_train, Y_train)
    print("finish fit")

    score = rf.score(X_test, Y_test)

    print(score)

    scores = cross_val_score(
        rf, X_train, Y_train, cv=5)

    print(scores.mean())

    Y_pred = rf.predict(X_test)

    from random import randrange

    Y_pred = []
    for i in Y_test:
        random_index = randrange(0, len(all_classes))
        Y_pred.append(all_classes[random_index])

    print(Y_pred)

    f1 = f1_score(Y_test, Y_pred, average='macro')

    print(f1)

    # pickle_write(rf, "rf.pkl")


train_test()
