import os
import cv2
import numpy as np
import torch


class load_image():

    def __init__(self, D, dims):

        if (D == 'Test'):
            dri = 'CUHK/dataset/testA/'
        elif (D == 'Train'):
            dri = 'CUHK/dataset/trainA/'
        elif (D == ''):
            dri = 'CUHK/dataset/testA/'

        self.img = []

        for f in os.listdir(dri):
            if not f.endswith('.jpg') and not endswith('.png'):
                continue
            # g= img_dir+f
            dim = dims
            # dim = (28,28)
            g = os.path.join(dri, f)
            im = cv2.imread(g, 0)
            imr = cv2.resize(im, dim)
            imr = np.expand_dims(imr, 0)

            imr = torch.from_numpy(imr)

            self.img.append((imr, f))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]

# train_dataset = load_image(D='', dims=(100,28))

# print(type(train_dataset[0][0]))
# test_dataset = load_image(D='Test', dims=(28,28))
# ll = np.array(train_dataset)
# print(ll[0][0].shape)


