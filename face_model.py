import os
import cv2
import encoder
from skimage.feature import *
from skimage.filters import *
from skimage.metrics import *
from sklearn.metrics.pairwise import cosine_similarity as coss
from skimage.io import *
import numpy as np
import matplotlib.pyplot as plt
# import torch


def load_img_read (T):
    if (T == 'Train'):
        img_dir='Dataset/CUHK/Training(88)/photo/'
    elif (T == 'Test'):
        img_dir='Dataset/CUHK/Testing(100)/photo/'
    elif (T == 'Test_sketch'):
        img_dir='Dataset/CUHK/Testing(100)/sketch/'


    filename = []
    images = []

    for f in sorted(os.listdir(img_dir)):
        # if not f.endswith('.jpg') and not endswith('.png'):
        if not f.endswith('.jpg'):
            continue
        # print(f)
        g= img_dir+f
        im = cv2.imread(g, 0)
        im = cv2.resize(im, (380, 350))
        images.append(im)
        filename.append(f)

    return images, filename

img, file_name = load_img_read(T='Test')
img_skt, file_name_skt = load_img_read(T='Test_sketch')

threshold = float(0.8)

for r in range(len(img)):
    lbps = local_binary_pattern(img[r], P=1, R=1, method='uniform')
    dogg = difference_of_gaussians(img[r], low_sigma=0.3, high_sigma=9)
    hog_feats, hog_img = hog(dogg, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                             multichannel=False,
                             block_norm='L2')

    lbps_skt = local_binary_pattern(img_skt[r], P=1, R=1, method='uniform')
    dogg_skt = difference_of_gaussians(img_skt[r], low_sigma=0.3, high_sigma=9)
    hog_feats_skt, hog_img_skt = hog(dogg_skt, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                     visualize=True, multichannel=False,
                                     block_norm='L2')


    print('SSIM', r, '---', structural_similarity(img[5], dogg+lbps))
    if float((structural_similarity(img[5], dogg))) >= threshold:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        ax1.imshow(img[5], cmap=plt.cm.gray)
        ax1.axis('off')
        ax1.set_title('original image', fontsize=20)

        ax2.imshow(dogg, cmap=plt.cm.gray)
        ax2.axis('off')
        ax2.set_title(r'DOG', fontsize=20)

        fig.tight_layout()
        plt.show()

        break
    # print('Cosine', r, '---', coss(dogg_skt, dogg))
    # print('------------------------------------------------------------------------')
