import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from dataset2 import *
import face_alignment
from torch.autograd import Variable


# model = torch.load('self_cycle_199.pth')
model.load_state_dict(torch.load('self_cycle_199.pth'))


# Image transformations
transforms_ = [ transforms.Resize((128, 128), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


# Training data loader
batch_size = 1
crd = os.getcwd()
root = os.path.join(crd, 'Dataset', 'AR Data', 'cropped')
print(os.listdir(root))

dataloader = DataLoader(
    ImageDataset(root, transforms_=transforms_, unaligned=True),
    batch_size=batch_size,
    shuffle=False, num_workers=1
)

print(len(dataloader))


# Validation data loader
val_dataloader = DataLoader(
    ImageDataset(root, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=batch_size,
    shuffle=False, num_workers=1
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    fake_B = self_cycle.G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = self_cycle.G_BA(real_B)
    img_sample = torch.cat((real_A.data, fake_B.data,
                            real_B.data, fake_A.data), 0)
    save_image(img_sample, 'B_U_images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# fake_A_ = fake_A_buffer.push_and_pop(Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width) * 0.0)
# fake_B_ = fake_B_buffer.push_and_pop(Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width) * 0.0)