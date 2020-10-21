import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
from smartimage.models.classify_3d.auto_vgg import Features
os.environ['VXM_BACKEND'] = 'pytorch'


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
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, label_model, label_target):
        x = torch.cat((label_model, label_target), dim=1)
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # x = F.grid_sample(x, grid)

        return theta

    def forward(self, label_model, label_target):
        # transform the input
        theta = self.stn(label_model, label_target)
        grid = F.affine_grid(theta, label_target.size())
        x = F.grid_sample(label_model, grid)
        # # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return x

# IMAGE_SIZE = 28*3
IMAGE_SIZE = 128
from voxelmorph.torch.networks import MySTN2d as MySTN
# class MySTN(nn.Module):
#     def __init__(self):
#         super(MySTN, self).__init__()
#         self.conv2_drop = nn.Dropout2d()
#
#         # Spatial transformer localization-network
#         self.localization = Features(input_size=(IMAGE_SIZE, IMAGE_SIZE), input_channel=2, conv_cfg=[16, "M", 16])
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(self.localization.output_features_number, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 2 * 3)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float))
#
#     # Spatial transformer network forward function
#     def stn(self, label_model, label_target):
#         x = torch.cat((label_model, label_target), dim=1)
#         xs = self.localization(x)
#         xs = xs.view(-1, self.localization.output_features_number)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#
#         # x = F.grid_sample(x, grid)
#
#         return theta
#
#     def forward(self, label_model, label_target):
#         # transform the input
#         theta = self.stn(label_model, label_target)
#         grid = F.affine_grid(theta, label_target.size())
#         x = F.grid_sample(label_model, grid)
#         # # Perform the usual forward pass
#         # x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         # x = x.view(-1, 320)
#         # x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=0.3, training=self.training)
#         # x = self.fc2(x)
#         return x

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
    net = torch.load('/private/medical-src2.x/ai-test/stn2.pth', map_location='cpu')
    # net = torch.load('/private/voxelmorph/processed_data/models/chest_label_stn/0129.pt', map_location='cpu')
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


    # grid = F.affine_grid(theta, torch.Size((1, 1, 28,28)))
    output = F.grid_sample(big_template, grid)
    output = output.detach().numpy()
    output = np.reshape(output, shape)

    return output

def transform(arr):
    new_arr = np.zeros_like(arr)
    small_arr = cv2.resize(arr, (180,180))
    new_arr[-180:, :180] = small_arr
    return new_arr



def main():
    import SimpleITK as sitk
    from matplotlib import pyplot as plt
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

    template = labels[30]
    target = labels[2]
    transformed_label = predict(template, target)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(template)
    axarr[0].set_title('template')

    axarr[1].imshow(target)
    axarr[1].set_title('real label')

    axarr[2].imshow(transformed_label)
    axarr[2].set_title('transformed_label')

    # plt.ioff()
    plt.show()
    print('finished')



if __name__ == "__main__":
    main()