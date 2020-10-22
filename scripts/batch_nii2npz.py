import SimpleITK as sitk
import numpy as np
import os


def normlize_nii():
    # dataset_path = '/private/voxelmorph/processed_data/image_3d_to_2d'
    dataset_path = '/private/voxelmorph/processed_data/2dlabels'
    # output_path = '/private/voxelmorph/processed_data/chest_train_images_npz_norm'
    output_path = '/private/voxelmorph/processed_data/chest_train_labels_npz_norm'
    normallize = True
    files = os.listdir(dataset_path)
    for i, f in enumerate(files):
        file_path = os.path.join(dataset_path, f)
        print(file_path)
        label = sitk.ReadImage(file_path)
        label_arr = sitk.GetArrayFromImage(label)
        label_arr = label_arr.squeeze()
        if normallize:
            label_arr = (label_arr - label_arr.mean())/label_arr.std()
        np.savez_compressed(os.path.join(output_path, f.replace('.nii.gz', '.npz')), vol=label_arr.astype(np.float32))


def npz2nii():
    input_file = '/private/voxelmorph/processed_data/output/A002323123_niimoved.npz'
    output_file = '/private/voxelmorph/processed_data/output/A002323123_niimoved.nii.gz'
    plabel_arr = np.load(input_file)['vol']
    int_arr = plabel_arr.astype(np.uint8)
    image = sitk.GetImageFromArray(int_arr)
    sitk.WriteImage(image, output_file)
    print(plabel_arr.max(), plabel_arr.min())


def nii2nii():
    dataset_path = '/private/voxelmorph/processed_data/train'
    output_path = '/private/voxelmorph/processed_data/train_image_nii'
    folders = os.listdir(dataset_path)
    for i, folder in enumerate(folders):
        dir_path = os.path.join(dataset_path, folder)
        src = os.path.join(dir_path, 'images_1.nii.gz')
        dst = os.path.join(output_path, folder+'.nii.gz')
        os.link(src, dst)

import torch
import torch.nn.functional as F
import os
os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph.torch.layers import SpatialTransformer
import voxelmorph as vxm

def grid_sample(image_arr_3d, grid_arr_4d):
    """

    :param image_arr_3d: (z,y,x)
    :param grid_arr_4d: (3,z,y,x)
    :return:
    """
    image_shape = image_arr_3d.shape
    grid_arr_5d = np.reshape(grid_arr_4d, (1, *grid_arr_4d.shape))
    image_arr_5d = np.reshape(image_arr_3d, (1, 1, *image_arr_3d.shape))
    grid_tensor = torch.tensor(grid_arr_5d.astype(np.float32))

    transform = SpatialTransformer(size=image_shape)
    # grid_tensor = grid_tensor.permute(0, 2, 3, 4, 1)
    # grid_tensor = grid_tensor[..., [2, 1, 0]]
    image_tensor = torch.tensor(image_arr_5d.astype(np.float32))
    output = transform(image_tensor, grid_tensor)
    output = output.detach().numpy()
    output = np.reshape(output, image_shape)
    return output

def make_grid_image(ref_image_path, output_grid_image_path):
    import SimpleITK as sitk

    ref_image = sitk.ReadImage(ref_image_path)
    grid_filter = sitk.GridImageSource()
    grid_filter.SetSize(ref_image.GetSize())
    grid_image = grid_filter.Execute()
    set_space(grid_image, ref_image)
    sitk.WriteImage(grid_image, output_grid_image_path)


def registration_by_warp():

    device = torch.device('cpu')
    # moving_image_path = '/private/voxelmorph/processed_data/output/grid.nii.gz'
    # moving_image_path = '/private/voxelmorph/processed_data/output/images_1.nii.gz'
    # ref_image_path = '/private/voxelmorph/processed_data/output/images_2.nii.gz'
    # grid_arr_path = '/private/voxelmorph/processed_data/output/1to2warp_bi.npz'
    # output_grid_image_path = '/private/voxelmorph/processed_data/output/images_2_grid.nii.gz'

    # moving_image_path = '/private/voxelmorph/processed_data/output/a001Normal0-labels__1__alone-30.nii.gz'
    # output_grid_image_path = '/private/voxelmorph/processed_data/output/a001Normal0-labels__1__alone-30to1_by_warp.nii.gz'

    moving_image_path = '/private/voxelmorph/processed_data/output/a001Normal0-images_1-30.nii.gz'
    output_grid_image_path = '/private/voxelmorph/processed_data/output/a001Normal0-images_1-30to1_by_warp.nii.gz'

    ref_image_path = '/private/voxelmorph/processed_data/2dlabels/a001Normal0-labels__1__alone-1.nii.gz'
    grid_arr_path = '/private/voxelmorph/processed_data/output/chest30to1warp_bi.npz'
    # grid_arr_path = '/private/voxelmorph/processed_data/output/chest_image_30to1_warp.npz'
    moving= vxm.py.utils.load_volfile(moving_image_path, add_batch_axis=True, add_feat_axis=True)
    fixed, fixed_affine = vxm.py.utils.load_volfile(ref_image_path, add_batch_axis=True, add_feat_axis=True,
                                                    ret_affine=True)
    image_shape = moving.shape[1:]
    if moving.ndim == 5:
        image_tensor = torch.from_numpy(moving.astype(np.float32)).float().permute(0, 4, 1, 2, 3).to(device)
    elif moving.ndim == 4:
        image_tensor = torch.from_numpy(moving.astype(np.float32)).float().permute(0, 3, 1, 2).to(device)
    grid_arr_4d = np.load(grid_arr_path)['vol']
    grid_arr_5d = np.reshape(grid_arr_4d, (1, *grid_arr_4d.shape))
    grid_tensor = torch.tensor(grid_arr_5d.astype(np.float32)).to(device)

    transform = SpatialTransformer(size=image_shape[:-1])
    transform.to(device)
    transform.eval()
    output = transform(image_tensor, grid_tensor)
    moved = output.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, output_grid_image_path, fixed_affine)


def cascade_warp():
    device = torch.device('cpu')
    moving_image_path = '/private/voxelmorph/processed_data/output/a001Normal0-images_1-30.nii.gz'
    output_grid_image_path = '/private/voxelmorph/processed_data/output/a001Normal0-images_1-30to1_by_warp.nii.gz'

    ref_image_path = '/private/voxelmorph/processed_data/2dlabels/a001Normal0-labels__1__alone-1.nii.gz'
    grid_arr_path = '/private/voxelmorph/processed_data/output/chest30to1warp_bi.npz'
    grid_arr_path2 = '/private/voxelmorph/processed_data/output/chest30to1warp_downsize8.npz'
    moving= vxm.py.utils.load_volfile(moving_image_path, add_batch_axis=True, add_feat_axis=True)
    fixed, fixed_affine = vxm.py.utils.load_volfile(ref_image_path, add_batch_axis=True, add_feat_axis=True,
                                                    ret_affine=True)
    image_shape = moving.shape[1:]
    if moving.ndim == 5:
        image_tensor = torch.from_numpy(moving.astype(np.float32)).float().permute(0, 4, 1, 2, 3).to(device)
    elif moving.ndim == 4:
        image_tensor = torch.from_numpy(moving.astype(np.float32)).float().permute(0, 3, 1, 2).to(device)

    grid_arr_4d = np.load(grid_arr_path)['vol']
    grid_arr_5d = np.reshape(grid_arr_4d, (1, *grid_arr_4d.shape))
    grid_tensor = torch.tensor(grid_arr_5d.astype(np.float32)).to(device)

    grid_arr_4d2 = np.load(grid_arr_path2)['vol']
    grid_arr_5d2 = np.reshape(grid_arr_4d2, (1, *grid_arr_4d.shape))
    grid_tensor2 = torch.tensor(grid_arr_5d2.astype(np.float32)).to(device)

    transform = SpatialTransformer(size=image_shape[:-1])
    transform.to(device)
    transform.eval()
    output = transform(image_tensor, grid_tensor)
    output = transform(output, grid_tensor2)
    moved = output.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, output_grid_image_path, fixed_affine)


def set_space(image, ref_image):
    image.SetOrigin(ref_image.GetOrigin())
    image.SetSpacing(ref_image.GetSpacing())
    image.SetDirection(ref_image.GetDirection())


if __name__ == "__main__":
    # nii2nii()
    # registration_by_warp()
    cascade_warp()
    # make_grid_image()
    # normlize_nii()