import os
import cv2
import numpy as np
# import torch
from skimage.metrics import structural_similarity, mean_squared_error, normalized_root_mse
from skimage.feature import hog, local_binary_pattern
from skimage.filters import difference_of_gaussians
from sklearn.metrics.pairwise import manhattan_distances

# in_img = 'f-039-01.jpg'
in_img = 'f-039-01.jpg.jpg'
# in_img = 'f-040-01.jpg.jpg'
# in_img = 'f-040-01.jpg'
# in_img2 = 'photo1.jpg'
# in_img3 = '007.png'
# in_img4 = 'OutputInpaint_new.png'
a = cv2.imread(in_img)
input_dog = difference_of_gaussians(a, low_sigma=0.3, high_sigma=9)
input_dog = input_dog.reshape(-1)
# b = cv2.imread(in_img2)
# b = cv2.imread(in_img2)
# c = cv2.imread(in_img3)
# d = cv2.imread(in_img4)


def matched_orb(img1, img2):
    bf = cv2.BFMatcher()
    match = bf.knnMatch(img1, img2, k=2)
    good_match = []
    for m, n in match:
        if m.distance < 0.75*n.distance:
            good_match.append(m)
    return  len(good_match)

cwd = os.listdir(os.getcwd()+'/Orb/test_photo')
ax = np.load(os.getcwd()+'/Orb/test_photo/'+cwd[0], allow_pickle=True)

orb = cv2.ORB_create()
kp2, a = orb.detectAndCompute(a, None)

best_t = []
imv = []
indx = 0
for b in ax:
    chk = matched_orb(a, b)
    if chk > 25:
        best_t.append(chk)
        imv.append(indx)
    indx += 1

# print(best_t)
# print(imv)

from scipy.spatial import distance

# import math
def measure_feat(img):
    inbuilt_dog2 = np.load('Dog/test_photo/featureTest_photo.npy')
    maxx = 0
    max_chk = 0
    indx = 0
    extract_img = []
    for a in inbuilt_dog2:
        # print(img.shape)
        # print(a.shape)
        # max_chk = max(maxx, structural_similarity(img, a))
        print('befe', maxx)
        max_chk = max(maxx, distance.euclidean(img, a))
        print('max_chk', max_chk)
        if max_chk < maxx:
            print('max_chk', max_chk)
            maxx = max_chk
            # print(indx)
            extract_img.append(indx)
        indx += 1

    return max_chk, extract_img

ac, av = measure_feat(input_dog)

print(ac)
print(av)
# print(len(ac), len(av))
