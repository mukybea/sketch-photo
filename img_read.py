import os
import cv2
# from skimage.feature import hog, local_binary_pattern
# from skimage.filters import difference_of_gaussians
# from skimage.metrics import structural_similarity, mean_squared_error, normalized_root_mse
# from sklearn.metrics.pairwise import cosine_similarity as coss
# from skimage.io import *
from skimage.filters import difference_of_gaussians
import numpy as np
# import sifft
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from PIL import Image
# import dlib
# import torchvision.transforms as transforms
import face_alignment
# import fis


# def load_img_read(T, save_c=None):
def load_img_read(T):
    if (T == 'Train_photo'):
        # img_dir = 'Dataset/CUHK/Training(88)/photo/'
        img_dir = 'aligned_images/Train/photo'
    elif (T == 'Train_sketch'):
        # img_dir = 'Dataset/CUHK/Training(88)/sketch/'
        img_dir = 'aligned_images/Train/sketch'
    elif (T == 'Test_photo'):
        # img_dir = 'Dataset/CUHK/Testing(100)/photo/'
        img_dir = 'aligned_images/Test/photo'
    elif (T == 'Test_sketch'):
        # img_dir = 'Dataset/CUHK/Testing(100)/sketch/'
        img_dir = 'aligned_images/Test/sketch'
    # else:
    #     img_dir = T


    filename = []
    images = []
    # p = 0
    # if save_c == 'Train_photo':
    #     save_path = 'aligned_images/Train/photo'
    # elif save_c == 'Train_sketch':
    #     save_path = 'aligned_images/Train/sketch'
    # elif save_c == 'Test_photo':
    #     save_path = 'aligned_images/Test/photo'
    # elif save_c == 'Test_sketch':
    #     save_path = 'aligned_images/Test/sketch'
    # else:
    #     save_path = save_c

    for f in sorted(os.listdir(img_dir)):
        # if p >= 2:
        #     break
        # if not f.endswith('.jpg') and not endswith('.png'):
        if not f.endswith('.jpg'):
            continue
        g = img_dir + '/'+f

        # p += 1
        # im1 = face_alignment.align_img(g)
        im = cv2.imread(g, 1)
        # print(g)
        # im = im.resize((192, 224), Image.ANTIALIAS)
        # lol = cv2.imwrite(os.path.join(save_path,f+'.jpg'), im1)
        # print(os.getcwd())
        images.append(im)
        filename.append(f)

    return images, filename


def measure_SSIM_MSE(img, img_skt):
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

        print('MSE DOG', r, '---', mean_squared_error(dogg_skt, dogg))
        print('MSE HOG', r, '---', mean_squared_error(hog_img_skt, hog_img))
        print('SSIM with LBP+HOG+DOG', r, '---',
              structural_similarity(lbps_skt + hog_img_skt + dogg_skt, lbps + hog_img + dogg)*100)
        print('SSIM with DOG', r, '---', structural_similarity(img[r], dogg)*100)

        if float((structural_similarity(img[r], dogg))) >= threshold:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            #
            ax1.imshow(img[r], cmap=plt.cm.gray)
            ax1.axis('off')
            ax1.set_title('original image', fontsize=20)

            ax2.imshow(dogg, cmap=plt.cm.gray)
            ax2.axis('off')
            ax2.set_title(r'DOG', fontsize=20)
            #
            fig.tight_layout()
            plt.show()

            break
        # print('Cosine', r, '---', coss(dogg_skt, dogg))
        print('------------------------------------------------------------------------')


def extract_feat(imge, name=None, feat_kind='Hog'):
        q = 0
        print(len(imge))
        feats_save = []
        for a in imge:
            # if feat_kind == 'Hog':
            #     feats, imgs = hog(a, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
            #                       multichannel=False,
            #                       block_norm='L2')
            #     feats_save.append(feats)
            # if feat_kind == 'Sift':
            #     sift = cv2.xfeatures2d.SIFT_create(100)
            #     _, feats = sift.detectAndCompute(a, None)
            #     feats = feats.reshape(-1)
            if feat_kind == 'Orb':
                orb = cv2.ORB_create()
                kp1, feats = orb.detectAndCompute(a, None)

                feats_save.append(feats)

            # elif feat_kind =='Lbp':
            #     feats = local_binary_pattern(a, P=1, R=1, method='uniform')
            #     feats = feats.reshape(-1)
            #     feats_save.append(feats)
            #
            elif feat_kind =='Dog':
                feats = difference_of_gaussians(a, low_sigma=0.3, high_sigma=9)
                feats = feats.reshape(-1)
                feats_save.append(feats)

        if name == 'Test_photo':
            if os.path.isdir(feat_kind+"/test_photo"):
                np.save(feat_kind+'/test_photo/feature'+name, feats_save)
                # np.save(feat_kind+'/test_photo/_feature_'+name+str(q), feats)
            else:
                os.makedirs(feat_kind+'/test_photo')
                np.save(os.path.join(feat_kind, 'test_photo','_feature_'+name), feats_save)
        elif name == 'Test_sketch':
            if os.path.isdir(feat_kind+"/test_sketch"):
                np.save(feat_kind+'/test_sketch/_feature_' + name, feats_save)
            else:
                os.makedirs(feat_kind+'/test_sketch')
                np.save(os.path.join(feat_kind, 'test_sketch', '_feature_' + name), feats_save)
        elif name == 'Train_photo':
            if os.path.isdir(feat_kind+"/train_photo"):
                np.save(feat_kind+'/train_photo/_feature_' + name, feats_save)
            else:
                os.makedirs(feat_kind+'/train_photo')
                np.save(os.path.join(feat_kind, 'train_photo', '_feature_' + name), feats_save)
        elif name == 'Train_sketch':
            if os.path.isdir(feat_kind+"/train_sketch"):
                np.save(feat_kind+'/train_sketch/_feature_' + name, feats_save)
            else:
                os.makedirs(feat_kind+'/train_sketch')
                np.save(os.path.join(feat_kind, 'train_sketch', '_feature_' + name), feats_save)

        q += 1



# img = load_img_read(T='Dataset/AR Data/sketch/', save_c='Dataset/AR Data/cropped/')
# imgi = face_alignment.align_img('Dataset/AR Data/sketch/')
# lol = cv2.imwrite(os.path.join('Dataset', 'AR Data', 'sketch'), imgi)


# print(fis.control_output(hog=0.38, sift=0.7, lbp=0.6, dog=0.87))

# img = np.load('Dog/test_photo/_feature_Test_photo0.npy')
# print(img)
# print(img.shape)
# imgi = cv2.imread('m-008-01.jpg', 1)
# imgi = face_alignment.align_img('Dataset/AR Data/sketch 2/Wp-057-1-sz1.jpg')
# cv2.imshow('ss', imgi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# lol = cv2.imwrite(os.path.join('Dataset/AR Data/cropped', 'Wp-057-1-sz1.jpg'), imgi)
# img2 = cv2.imread('OutputInpaint_new_1.png', 0)
# imgi = cv2.imread('sketch2.jpg', 0)
# img2 = cv2.imread('OutputInpaint_new_24.png', 0)
# imgi = cv2.resize(imgi, (64, 64))
# img2 = cv2.resize(img2, (64, 64))


# feats, imgix = hog(imgi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
#                                   multichannel=True,
#                                   block_norm='L2')


# feats = difference_of_gaussians(imgi, low_sigma=0.3, high_sigma=9)
# feats = feats.reshape(-1)


# print(img.shape)
# print(feats.shape)
import math
# ll = [[0]*len(img)]*len(img)
# ll = [float]*len(feats)
# print(ll[:4])
# ssum = 0
# for i in range(len(feats)):
#     ll[i] = math.sqrt(math.pow((img[i] - feats[i]), 2))
#     ssum += ll[i]
# print(ssum)

# for i in range(len(img[0])):
#     for j in range(len(img[1])):
#         # print(imgi[i][6])
#         # print(img2[i][6])
#         # print(float(img2[i][j] - imgi[i][j]))
#         ll[i][j] = math.sqrt(math.pow((img[i][j] - imgix[i][j]),2))
#         ssum += ll[i][j]
# ll = (img, feats)

# print(mean_squared_error(feats, img))
# print(normalized_root_mse(feats, img))
# print(structural_similarity(feats, img))
# imgi = face_alignment.align_img('f-039-01.jpg')
# imgi = cv2.resize(imgi, (128,128))
# print(imgi.shape)
# b, g, r = cv2.split(imgi)
# image_rgb = cv2.merge([r, g, b])
# plt.imshow(image_rgb)
# plt.imshow(imgi)
# plt.show()
# img2 = cv2.imread('OutputInpaint_new.png')
# print(img2.shape)
# print('SSIM', structural_similarity(imgi, img2, multichannel=True))
# cv2.imshow('j', imgi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# imgi = cv2.resize(imgi, (128, 64), Image.ANTIALIAS)
# feats, _ = hog(imgi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False, block_norm='L2')
# cc = cv2.xfeatures2d.SIFT_create(400)
# cc = cv2.xfeatures2d.SURF_create()
# cc = cv2.xfeatures2d.SURF_create(hessianThreshold=200)

# print(cc.hessianThreshold)
# print(imgi.shape)
# b, a = cc.detectAndCompute(imgi, None)
# img2 = cv2.drawKeypoints(imgi,b,None,(255,0,0),2)
# feats = local_binary_pattern(imgi, P=1, R=1, method='uniform')
# feats = difference_of_gaussians(imgi, low_sigma=0.3, high_sigma=9)
# plt.imshow(img2),plt.show()

# print(feats.shape)
# print(feats.reshape(-1).shape)
# print(a.reshape(-1).shape)
# print(a.shape)

# import cv2
# img_photo_test, filename_photo_test = load_img_read(T='Test_photo', save_c='Test_photo')
img_photo_test, filename_photo_test = load_img_read(T='Test_photo')
img_skt_test, filename_skt_test = load_img_read(T='Test_sketch')
# img_skt_test, filename_skt_test = load_img_read(T='Test_sketch', save_c='Test_sketch')
# #######################################
# img_photo_train, filename_photo_train = load_img_read(T='Train_photo', save_c='Train_photo')
img_photo_train, filename_photo_train = load_img_read(T='Train_photo')
img_skt_train, filename_skt_train = load_img_read(T='Train_sketch')
# img_skt_train, filename_skt_train = load_img_read(T='Train_sketch', save_c='Train_sketch')
#
# #Hog
# extract_hog_feat(img_skt_train, name='Train_sketch', feat_kind='Hog')
# extract_hog_feat(img_photo_train, name='Train_photo', feat_kind='Hog')
# extract_hog_feat(img_skt_test, name='Test_sketch', feat_kind='Hog')
# extract_hog_feat(img_photo_test, name='Test_photo', feat_kind='Hog')
# # #Orb
# extract_feat(img_skt_train, name='Train_sketch', feat_kind='Orb')
# extract_feat(img_photo_train, name='Train_photo', feat_kind='Orb')
# extract_feat(img_skt_test, name='Test_sketch', feat_kind='Orb')
# extract_feat(img_photo_test, name='Test_photo', feat_kind='Orb')
# # #Surf
# # extract_hog_feat(img_skt_train, name='Train_sketch', feat_kind='Surf')
# # extract_hog_feat(img_photo_train, name='Train_photo', feat_kind='Surf')
# # extract_hog_feat(img_skt_test, name='Test_sketch', feat_kind='Surf')
# # extract_hog_feat(img_photo_test, name='Test_photo', feat_kind='Surf')
# # #Lbp
# extract_hog_feat(img_skt_train, name='Train_sketch', feat_kind='Lbp')
# extract_hog_feat(img_photo_train, name='Train_photo', feat_kind='Lbp')
# extract_hog_feat(img_skt_test, name='Test_sketch', feat_kind='Lbp')
# extract_hog_feat(img_photo_test, name='Test_photo', feat_kind='Lbp')
# # #Dog
extract_feat(img_skt_train, name='Train_sketch', feat_kind='Dog')
extract_feat(img_photo_train, name='Train_photo', feat_kind='Dog')
extract_feat(img_skt_test, name='Test_sketch', feat_kind='Dog')
extract_feat(img_photo_test, name='Test_photo', feat_kind='Dog')
# #


