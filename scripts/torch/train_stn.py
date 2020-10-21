#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import random
import argparse
import glob
import time
import numpy as np
import torch

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from voxelmorph.torch.networks import MySTN2d as MySTN
from smartimage.metrics.metrics import DiceLoss2D


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--datadir', help='base data directory')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
parser.add_argument('--regex', default='*.nii.gz', help='suffix of file')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet',  action='store_true', help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

# 默认参数
args.datadir = '/private/voxelmorph/processed_data/2dlabels'
args.model_dir = '/private/voxelmorph/processed_data/models/chest_label_stn'
args.epochs = 200
args.regex = '*.nii.gz'
args.image_loss = 'dice'
args.lr = 0.001

bidir = args.bidir
print(args.regex)
# load and prepare training data
train_vol_names = glob.glob(os.path.join(args.datadir, args.regex))
random.shuffle(train_vol_names)  # shuffle volume list
assert len(train_vol_names) > 0, 'Could not find any training data'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_vol_names, atlas, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(train_vol_names, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert args.batch_size >= nb_gpus, 'Batch size (%d) should be no less than the number of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = torch.load(args.load_model, map_location=device)
else:
    # otherwise configure new model
    model = MySTN(inshape=inshape, conv_cfg=[16, 32, 64, 'M', 128, 128, 'M', 64, 64, 'M',]).to(device)

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
elif args.image_loss == 'dice':
    image_loss_func = DiceLoss2D()
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses  = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses  = [image_loss_func]
    weights = [1]

# prepare deformation loss
# losses  += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
# weights += [args.weight]

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    torch.save(model, os.path.join(model_dir, '%04d.pt' % epoch))

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        if len(inshape) == 3:
            inputs = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        elif len(inshape) == 2:
            inputs = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d.astype(np.float32)).to(device).float().permute(0, 3, 1, 2) for d in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        # for n, loss_function in enumerate(losses):
            # curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
        curr_loss = losses[0](y_true[0], y_pred)
        loss_list.append('%.6f' % curr_loss.item())
        loss += curr_loss

        loss_info = 'loss: %.6f  (%s)' % (loss.item(), ', '.join(loss_list))

        # backpropagate and optimize
        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

        # print step info
        epoch_info = 'epoch: %04d' % (epoch + 1)
        step_info = ('step: %d/%d' % (step + 1, args.steps_per_epoch)).ljust(14)
        time_info = 'time: %.2f sec' % (time.time() - step_start_time)
        print('  '.join((epoch_info, step_info, time_info, loss_info)), flush=True)

# final model save
# model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))