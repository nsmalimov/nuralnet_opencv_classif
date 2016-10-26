import sys

sys.path.append('/usr/local/lib/python2.7/site-packages/')

import cv2
import numpy as np

print np.__file__
print cv2.__file__

print(cv2.__version__)


def get_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    (kps, descs) = sift.detectAndCompute(gray, None)

    return descs.flatten()


res = get_features('/cifar10_opencv/lubuntu-logo.png')

print(res)
