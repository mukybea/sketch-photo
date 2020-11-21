from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import torch.optim as optim

# import pytorch_ssim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import copy
import numpy as np
import img_read
import face_alignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

def image_loader(image_name):
    # size = (414, 582)
    # image = Image.open(image_name)
    # image = image.resize([414,582], Image.ANTIALIAS)
    # image = image_name.resize([414,582], Image.ANTIALIAS)
    # print(image.size)
    # fake batch dimension required to fit network's input dimensions
    # image = loader(image).unsqueeze(0)
    # imm = imshow(image_name)
    image = loader(image_name).unsqueeze(0)
    # print(image.shape)
    return image.to(device, torch.float)


def imshow(tensor, title=None, save=None, bgr_img=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)

    plt.imshow(image)
    if save is not None:
        print(type(image))
        print(image.size)
        image = image.save('OutputInpaint_new_25.png')
        # plt.imsave('OutputInpaint.png', image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor




#
unloader = transforms.ToPILImage()  # reconvert into PIL image
img, file_name = img_read.load_img_read(T='Test')
print(type(img))
img = np.array(img)
print(img.shape)
img = unloader(img)
print(img.shape)
print(image_loader(img).size())
# img_skt, file_name_skt = img_read.load_img_read(T='Test_sketch')
# im1 = face_alignment.align_img("Dataset/CUHK/Testing(100)/photo/"+file_name[1])
# im2 = face_alignment.align_img("Dataset/CUHK/Testing(100)/sketch/"+file_name_skt[1])

# unloader = transforms.ToPILImage()  # reconvert into PIL image

# style_img = image_loader("f-039-01.jpg")
# content_img = face_alignment.align_img('f-039-01-sz1.jpg')
content_img = face_alignment.align_img('f-039-01-sz1.jpg')
style_img = face_alignment.align_img('photo1.jpg')
# style_img = face_alignment.align_img('f-039-01.jpg')
# content_img = face_alignment.align_img('sketch2.jpg')

style_img = unloader(style_img)

content_img = unloader(content_img)
style_img = image_loader(style_img)
content_img = image_loader(content_img)
# style_img = Image.fromarray(np.uint8(style_img)*255)
# style_img = loader(style_img).unsqueeze(0)
# style_img= style_img.to(device, torch.float)

# im1 = unloader(im1)
# style_img = image_loader(im1)
# im2 = unloader(im2)
# content_img = image_loader(im2)


assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

# unloader = transforms.ToPILImage()  # reconvert into PIL image

# plt.ion()



plt.figure()
imshow(style_img, title='Style Image', save='sa')

plt.figure()
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=False).features.to(device).eval()
# cnn = models.vgg19(pretrained=True).to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    # if torch.load('style_trans_model_1.pth'):
    #     model = torch.load('style_trans_model_1.pth')
    #     print(model(content_img))
    #     exit()
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img)
    # print(model)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)
    torch.save(model.state_dict(), 'style_trans_model_1.pth')
    # if model.load_state_dict(torch.load('style_trans_model_1.pth')):
    #     print('Yes')
    return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image', save='Save')
# sphinx_gallery_thumbnail_number = 4
# plt.ioff()
plt.show()