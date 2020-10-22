import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
from voxelmorph.torch.networks import MySTN2d as MySTN

plt.ion()

import cv2

def grid_sample(image_arr_3d, grid_arr_4d):
    """

    :param image_arr_3d: (z,y,x)
    :param grid_arr_4d: (3,z,y,x)
    :return:
    """
    image_shape = image_arr_3d.shape
    grid_arr_5d = np.reshape(grid_arr_4d, (1, *grid_arr_4d.shape))
    image_arr_5d = np.reshape(image_arr_3d, (1,1,*image_arr_3d.shape))
    grid_tensor = torch.tensor(grid_arr_5d.astype(np.float32))
    grid_tensor = grid_tensor.permute(0, 2, 3, 4, 1)
    grid_tensor = grid_tensor[..., [2, 1, 0]]
    image_tensor = torch.tensor(image_arr_5d.astype(np.float32))
    output = F.grid_sample(image_tensor, grid_tensor)
    output = output.detach().numpy()
    output = np.reshape(output, image_shape)
    return output

def predict(template, target):

    with torch.no_grad():
        net = torch.load('/private/voxelmorph/processed_data/models/train_by_minist/size128.pth', map_location='cpu')
        IMAGE_SIZE = 128
        #
        # net = torch.load('/private/voxelmorph/processed_data/models/chest_label_stn/0199.pt', map_location='cpu')
        # IMAGE_SIZE = 128

        shape = target.shape
        tmp_template = cv2.resize(template, (IMAGE_SIZE, IMAGE_SIZE))
        target = cv2.resize(target, (IMAGE_SIZE, IMAGE_SIZE))
        tmp_template = np.reshape(tmp_template, (1, 1, IMAGE_SIZE, IMAGE_SIZE))
        target = np.reshape(target, (1, 1, IMAGE_SIZE, IMAGE_SIZE))
        tmp_template = torch.tensor(tmp_template.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        theta = net.stn(tmp_template, target)
        big_template = cv2.resize(template, template.shape)
        big_template = np.reshape(big_template, (1, 1, *shape))
        big_template = torch.tensor(big_template.astype(np.float32))

        grid = F.affine_grid(theta, torch.Size((1, 1, *shape)))
        output = F.grid_sample(big_template, grid)
        output = output.detach().numpy()
        output = np.reshape(output, shape)

    return output

def transform(arr):
    new_arr = np.zeros_like(arr)
    small_arr = cv2.resize(arr, (180,180))
    new_arr[-180:, :180] = small_arr
    return new_arr


def show_result(moving, fix, moved):

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(moving)
    axarr[0].set_title('moving')

    axarr[1].imshow(fix)
    axarr[1].set_title('fix')

    axarr[2].imshow(moved)
    axarr[2].set_title('moved')

    plt.ioff()
    plt.show()


def main2():
    label_image_1 = sitk.ReadImage('/private/voxelmorph/processed_data/output/triangle_label1.nii.gz')
    labels_1 = sitk.GetArrayFromImage(label_image_1)
    label_image_2 = sitk.ReadImage('/private/voxelmorph/processed_data/output/triangle_label2.nii.gz')
    labels_2 = sitk.GetArrayFromImage(label_image_2)
    moved = predict(labels_1[0], labels_2[0])
    show_result(labels_1[0], labels_2[0], moved)


def main():

    label_image = sitk.ReadImage('/data/medical-ai2/Seg2D/胸片分割/preprocess_data/all/a001Normal0/labels__1__alone.nii.gz')
    labels = sitk.GetArrayFromImage(label_image)

    # fake_label = transform(labels[0])
    # transformed_label = predict(labels[2], fake_label)
    # f, axarr = plt.subplots(1, 3)
    # axarr[0].imshow(labels[0])
    # axarr[0].set_title('template')
    #
    # axarr[1].imshow(fake_label)
    # axarr[1].set_title('fake_label')
    #
    # axarr[2].imshow(transformed_label)
    # axarr[2].set_title('transformed_label')

    template = labels[5]
    target = labels[36]
    moved = predict(template, target)
    show_result(template, target, moved)
    print('finished')


if __name__ == "__main__":
    main2()