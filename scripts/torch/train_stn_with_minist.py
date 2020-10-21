# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description : 
"""
# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_
.. figure:: /_static/img/stn/FSeq.png
In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__
Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.
One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from smartimage.metrics.metrics import DiceLoss2D

plt.ion()  # interactive mode

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
IMAGE_SIZE = 28*3
# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomResizedCrop(size=28, scale=(0.5, 1), interpolation=Image.NEAREST),
                       transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
                       transforms.ToTensor(),

                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=16, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.5, 1), interpolation=Image.NEAREST),
        transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))

    ])), batch_size=16, shuffle=True, num_workers=4)


######################################################################
# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    We need the latest version of PyTorch that contains
#    affine_grid and grid_sample modules.
#


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2_drop = nn.Dropout2d()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
            # nn.Linear(32, 6 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, label_model, label_target):
        x = torch.cat((label_model, label_target), dim=1)
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)

        return grid

    def forward(self, label_model, label_target):
        # transform the input
        grid = self.stn(label_model, label_target)
        x = F.grid_sample(label_model, grid)
        # # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        # x = self.fc2(x)
        return x

from voxelmorph.torch.networks import MySTN2d as MySTN
######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.
import cv2
from matplotlib import pyplot as plt


def generate_num_pic(num, shape):
    ret = np.zeros(shape, np.uint8)
    ret = cv2.putText(ret, str(num), (3*3, 25*3), cv2.FONT_HERSHEY_SIMPLEX, 1*3, (1,), 2*3)
    return ret


SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
num_pic_dict = {
    0: generate_num_pic(0, SHAPE),
    1: generate_num_pic(1, SHAPE),
    2: generate_num_pic(2, SHAPE),
    3: generate_num_pic(3, SHAPE),
    4: generate_num_pic(4, SHAPE),
    5: generate_num_pic(5, SHAPE),
    6: generate_num_pic(6, SHAPE),
    7: generate_num_pic(7, SHAPE),
    8: generate_num_pic(8, SHAPE),
    9: generate_num_pic(9, SHAPE),

}


def show_num():
    b = generate_num_pic(8, (28, 28))
    plt.imshow(b)
    plt.show()


def get_nums_batch(nums, shape):
    batch = np.zeros((len(nums), 1, shape[0], shape[1]), dtype=np.float32)
    for i, num in enumerate(nums):
        batch[i, 0, :, :] = num_pic_dict[num]
    return batch


def normalize_tensor_batch(t):
    normalizer = transforms.Normalize((0.1307,), (0.3081,))
    ret = torch.empty(t.shape, dtype=t.dtype)
    for i, b in enumerate(t):
        ret[i] = normalizer(b)
    return ret


def run_train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        label_model = torch.tensor(get_nums_batch(target.numpy(), SHAPE))
        # label_model = normalize_tensor_batch(label_model)
        # data = normalize_tensor_batch(data)
        label_target = data.to(device)
        label_model = label_model.to(device)
        optimizer.zero_grad()
        output = model(label_model, label_target)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, label_target)
        # print(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


#
# A simple test procedure to measure STN the performances on MNIST.
#


def run_test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        # correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            # data, target = data.to(device), target.to(device)

            label_model = torch.tensor(get_nums_batch(target.numpy(), SHAPE))
            # label_model = normalize_tensor_batch(label_model)
            # data = normalize_tensor_batch(data)

            label_target = data.to(device)
            label_model = label_model.to(device)

            output = model(label_model, label_target)

            # sum up batch loss
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            loss = loss_func(output, label_target)
            test_loss += loss
            # get the index of the max log-probability
            # pred = output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()
        print('Test Avg Loss: {:.6f}'.format(test_loss / batch_idx))

        # test_loss /= len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}\n'
        #       .format(test_loss))


######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data, target = next(iter(test_loader))
        label_target = data.to(device)
        label_model = torch.tensor(get_nums_batch(target.numpy(), SHAPE))
        label_model = label_model.to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model(label_model, label_target).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))
        # torchvision.utils.make_grid(label_model.cpu()))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


if __name__ == '__main__':
    model = MySTN(inshape=SHAPE).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_func = DiceLoss2D()
    for epoch in range(1, 20 + 1):
        run_train(epoch)
        run_test()
    torch.save(model, '/private/medical-src2.x/ai-test/stn2.pth')
    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.show()
