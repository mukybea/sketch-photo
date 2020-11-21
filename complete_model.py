import os
from skimage.feature import hog, local_binary_pattern
from skimage.filters import difference_of_gaussians
from skimage.metrics import structural_similarity, mean_squared_error, normalized_root_mse
import numpy as np
# import matplotlib.pyplot as plt
import face_alignment
# import fis


#Step 1 read image
# in_img2 = 'photo1.jpg'
# in_img3 = '007.png'
def read_img(img):
    input_image = face_alignment.align_img(in_img)
    return input_image
# a = read_img(in_img)
# b = read_img(in_img2)
# c = read_img(in_img3)
# sift = cv2.xfeatures2d.SIFT_create(100)
# _, feats = sift.detectAndCompute(a, None)
# _, feats2 = sift.detectAndCompute(b, None)
# _, feats3 = sift.detectAndCompute(c, None)
# feats = feats.reshape(-1)
# feat2 = feats2.reshape(-1)
# feat3 = feats2.reshape(-1)
# print(feats.shape)
# print(feat2.shape)
# print(feat3.shape)

#Step2 - extract all 4 features
def feat_extract(in_img):
    input_hog = hog(in_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False, block_norm='L2')

    input_sift = cv2.xfeatures2d.SIFT_create()
    _, feats = input_sift.detectAndCompute(in_img, None)
    input_sift = feats.reshape(-1)

    input_lbp = local_binary_pattern(in_img, P=1, R=1, method='uniform')
    input_lbp = input_lbp.reshape(-1)

    input_dog = difference_of_gaussians(in_img, low_sigma=0.3, high_sigma=9)
    input_dog = input_dog.reshape(-1)

    return input_hog, input_sift, input_dog, input_lbp

#Step 3 - load inbuilt features
inbuilt_dog = []
i = 0
for a in (os.listdir(os.getcwd()+'/Dog/train_photo')):
    print(a)
    inbuilt_dog.append((np.load(os.getcwd()+'/Dog/train_photo/'+a)))
    # print(inbuilt_dog[i].shape)
#     i+=1

print(len(inbuilt_dog[0]))
print(inbuilt_dog[0])
# np.save('inbuilt_dog_train_photo.npy', inbuilt_dog)

# inbuilt_dog2 = np.load(os.getcwd()+'/Orb/test_photo/featureTest_photo.npy', allow_pickle=True)
# print(len(inbuilt_dog2))
# print(inbuilt_dog2[89].shape)

def measure_feat(img):
    inbuilt_dog2 = np.load('inbuilt_dog_train_photo.npy')
    maxx = 0
    max_chk = 0
    extract_img = []
    for a in inbuilt_dog2:
        max_chk = max(mean_squared_error(img, a))
        if max_chk > maxx:
            maxx = max_chk
            extract_img.append(a)

    return max_chk, extract_img[-1]

