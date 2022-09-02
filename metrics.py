import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from make_loader import make_loader
from train_process import calc_loss, print_metrics

from transform import DualCompose, CenterCrop, ImageOnly, Normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_metrics(train_file_names, val_file_names, test_file_names, mask_dir, dataset, mean_values, std_values, model,
                 fold_out='0', fold_in='0', name_model='UNet11', epochs='40', out_file='VHR', dataset_file='VHR',
                 name_file='_VHR_60_fake'):
    outfile_path = 'predictions/{}/'.format(out_file)

    if not os.path.exists("predictions/{}".format(out_file)):
        os.mkdir("predictions/{}".format(out_file))

    f = open(
        "predictions/{}/metric{}_{}_foldout{}_foldin{}_{}epochs.txt".format(out_file, name_file, name_model, fold_out,
                                                                            fold_in, epochs), "w+")
    f2 = open(
        "predictions/{}/pred_loss_test{}_{}_foldout{}_foldin{}_{}epochs.txt".format(out_file, name_file, name_model,
                                                                                    fold_out, fold_in, epochs), "w+")
    f.write("Training mean_values:[{}], std_values:[{}] \n".format(mean_values, std_values))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(len(test_file_names))

    all_transform = DualCompose([
        CenterCrop(int(512)),
        ImageOnly(Normalize(mean=mean_values, std=std_values))
    ])

    train_loader = make_loader(filenames=train_file_names,
                                     mask_dir=mask_dir,
                                     dataset=dataset,
                                     shuffle=False,
                                     transform=all_transform,
                                     mode='train')

    val_loader = make_loader(filenames=val_file_names,
                                   mask_dir=mask_dir,
                                   dataset=dataset,
                                   shuffle=False,
                                   transform=all_transform,
                                   mode='train')

    test_loader = make_loader(filenames=test_file_names,
                                    mask_dir=mask_dir,
                                    dataset=dataset,
                                    shuffle=False,
                                    transform=all_transform,
                                    mode='train')

    dataloaders = {
        'train': train_loader, 'val': val_loader, 'test': test_loader}

    for phase in ['train', 'val', 'test']:
        model.eval()
        metrics = defaultdict(float)

        count_img = 0
        input_vec = []
        labels_vec = []
        pred_vec = []
        result_dice = []
        result_jaccard = []

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                input_vec.append(inputs.data.cpu().numpy())
                labels_vec.append(labels.data.cpu().numpy())
                pred = model(inputs)

                loss = calc_loss(pred, labels, metrics, 'test')

                if phase == 'test':
                    print_metrics(metrics, f2, 'test')

                pred = torch.sigmoid(pred)
                pred_vec.append(pred.data.cpu().numpy())

                result_dice += [metrics['dice']]

                result_jaccard += [metrics['jaccard']]

                count_img += 1

        print("{}_{}".format(phase, out_file))
        print('Dice = ', np.mean(result_dice), np.std(result_dice))
        print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard), '\n')

        f.write("{}_{}\n".format(phase, out_file))
        f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice), np.std(result_dice)))
        f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard), np.std(result_jaccard)))

        if phase == 'test':
            np.save(str(os.path.join(outfile_path,
                                     "inputs_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file, name_model,
                                                                                                  fold_out, fold_in,
                                                                                                  epochs,
                                                                                                  int(count_img)))),
                    np.array(input_vec))
            np.save(str(os.path.join(outfile_path,
                                     "labels_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file, name_model,
                                                                                                  fold_out, fold_in,
                                                                                                  epochs,
                                                                                                  int(count_img)))),
                    np.array(labels_vec))
            np.save(str(os.path.join(outfile_path,
                                     "pred_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file, name_model,
                                                                                                fold_out, fold_in,
                                                                                                epochs,
                                                                                                int(count_img)))),
                    np.array(pred_vec))