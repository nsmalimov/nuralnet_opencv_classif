import pickle

import cv2

path_to_train = "/cifar10_images/train/"

path_to_test = "/cifar10_images/test/"

filename_train_labels = "/cifar10_images/trainLabels.csv"


def read_train_data_csv(filename_train_labels):
    f = open(filename_train_labels)

    data_dict = {}

    first = True
    for i in f.readlines():

        if (first):
            first = False
            continue

        s = i.replace("\n", "")
        s = s.split(",")
        data_dict[s[0]] = s[1]

    return data_dict


def get_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    (kps, descs) = sift.detectAndCompute(gray, None)

    return descs


def matrix_to_vec(matrix):
    vec = matrix.flatten()
    return vec


def get_image_features(filespath, test_flag=False):
    from os import listdir
    from os.path import isfile, join
    images_pth_list = [f for f in listdir(filespath) if isfile(join(filespath, f))]

    len_arr = []

    num = 0
    for i in images_pth_list:
        features_list = get_features(filespath + i)

        if (features_list is None):
            continue

        features_list = matrix_to_vec(features_list)

        len_arr.append(len(features_list))

        if (num % 100 == 0):
            print(num)

        num += 1

    print
    max(len_arr)

    exit()

    data_dict = {}

    num = 0
    for i in images_pth_list:
        image_num = i.replace(".png", "")
        features_list = get_features(filespath + i)

        if (features_list is None):
            continue

        features_list = matrix_to_vec(features_list)

        if (not (test_flag)):
            data_dict[image_num] = [features_list, train_labels[image_num]]
        else:
            data_dict[image_num] = features_list

            if (num >= 10000):
                break

        if (num % 100 == 0):
            print(num)

        num += 1

    return data_dict


def pickle_write(data, filename):
    f = open(filename, "w")

    pickle.dump(data, f)

    f.close()


train_labels = read_train_data_csv(filename_train_labels)

print(len(train_labels))

images_train = get_image_features(path_to_train, False)

pickle_write(images_train, "images_train.pkl")

# images_test = get_image_features(path_to_test, True)

# pickle_write(images_test, "images_test.pkl")

# for i in images_train:
#    print len(images_train[i][0])
