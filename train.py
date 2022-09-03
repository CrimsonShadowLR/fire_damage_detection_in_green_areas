import argparse
import glob
import os

import torch
from torch.backends import cudnn
import numpy as np
# python3 train.py --batch-size 4 --lr 1e-3  --n-epochs 5


from train_process import train_model
import dataset_utils
from make_loader import make_loader
from metrics import find_metrics
from preprocessing import meanstd
from transform import DualCompose, CenterCrop, HorizontalFlip, VerticalFlip, Rotate, ImageOnly, Normalize
from models.unet import UNet11


def train():
    """
    Training function
    Adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    &            https://github.com/csosapezo/socioeconomic-satellite-classification/
    """

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # image-related variables
    arg('--dataset-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/images', help='satellite image for LandSat8 directory')
    arg('--masks-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/masks', help='masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/split_npy', help='numPy preprocessed patches directory')

    # preprocessing-related variables val, test and the rest is train
    arg('--val-percent', type=float, default=0.15, help='Validation percent')
    arg('--test-percent', type=float, default=0.10, help='Test percent')

    # training-related variable
    arg('--batch-size', type=int, default=1, help='HR(High resolution):4,VHR(Very high resolution):8')
    arg('--limit', type=int, default=0, help='number of images in epoch')
    arg('--n-epochs', type=int, default=5)
    arg('--lr', type=float, default=1e-3)
    arg('--step', type=float, default=120)
    arg('--model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])
    arg('--target_segmentation', type=str, default='burn', help='burn: burn segmentation')
    arg('--out-path', type=str, default='./trained_models/', help='model output path')
    arg('--pretrained', type=int, default=1, help='0: False; 1: True')

    # CUDA devices
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')

    args = parser.parse_args()

    pretrained = True if args.pretrained else False

    model = UNet11(pretrained=pretrained)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        cudnn.benchmark = True

    images_filenames = np.array(sorted(glob.glob(args.dataset_dir + "/*.tif")))

    train_set_indices, val_set_indices, test_set_indices = dataset_utils.train_val_test_split(len(images_filenames),
                                                                                      args.val_percent,
                                                                                      args.test_percent)

    images_np_filenames = dataset_utils.save_npy(images_filenames, args.npy_dir, args.model, args.masks_dir)

    channel_num = 8
    _, mean_train, std_train = meanstd(np.array(images_np_filenames)[train_set_indices],
                                                     channel_num=channel_num)

    train_transform = DualCompose([
        # HorizontalFlip(),
        # VerticalFlip(),
        # Rotate(),
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    limit = args.limit if args.limit > 0 else None

    train_loader = make_loader(filenames=np.array(images_filenames)[train_set_indices],
                                     mask_dir=args.masks_dir,
                                     dataset=args.model,
                                     shuffle=False,
                                     transform=train_transform,
                                     mode='train',
                                     batch_size=args.batch_size,
                                     limit=limit)

    valid_loader = make_loader(filenames=np.array(images_filenames)[val_set_indices],
                                     mask_dir=args.masks_dir,
                                     dataset=args.model,
                                     shuffle=False,
                                     transform=val_transform,
                                     mode='train',
                                     batch_size=args.batch_size,
                                     limit=None)

    dataloaders = {
        'train': train_loader, 'val': valid_loader
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    name_file = '_' + str(int(args.val_percent * 100)) + '_percent_' + args.model

    train_model(name_file=name_file,
                      model=model,
                      dataset=args.model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      dataloaders=dataloaders,
                      name_model="Unet11",
                      num_epochs=args.n_epochs)

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    torch.save(model.module.state_dict(),
               (str(args.out_path) + '/model{}_{}_{}epochs').format(name_file, "Unet11", args.n_epochs))

    #find_metrics(train_file_names=np.array(images_np_filenames)[train_set_indices],
#                 val_file_names=np.array(images_np_filenames)[val_set_indices],
 #                test_file_names=np.array(images_np_filenames)[test_set_indices],
  #               mask_dir=args.masks_dir,
   #              dataset=args.model,
    #             mean_values=mean_train,
#                 std_values=std_train,
 #                model=model,
  #               name_model="Unet11",
   #              epochs=args.n_epochs,
    #             out_file=args.model,
#                 dataset_file=args.model,
 #                name_file=name_file)


if __name__ == "__main__":
    train()
