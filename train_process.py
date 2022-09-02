import os
import time
import torch
import torch.nn.functional as F
import copy
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


def print_metrics(metrics, file, phase='train', epoch_samples=1):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    if phase == 'test':
        file.write("{}".format(",".join(outputs)))
    else:
        print("{}: {}".format(phase, ", ".join(outputs)))
        file.write("{}: {}".format(phase, ", ".join(outputs)))


# The Jaccard coefficient measures similarity between finite sample sets.
def metric_jaccard(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    epsilon = 1e-15  # epsilon! para evitar el indeterminado
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = target.sum(dim=2).sum(dim=2) + pred.sum(dim=2).sum(dim=2) - intersection
    cjaccard = (intersection + epsilon) / (union + epsilon)
    loss = 1 - cjaccard
    return loss.mean()  # mean of the batch

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    cdice = (2. * intersection + smooth) / (
                pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)  # 1-Dice
    loss = 1 - cdice
    return loss.mean()  # mean of the batch

def calc_loss(pred, target, metrics, dataset, phase='train', bce_weight=0.3):

    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    # convering tensor to numpy to remove from the computationl graph
    if phase == 'test':
        pred = (pred > 0.50).float()  # with 0.55 is a little better
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] = loss.data.cpu().numpy() * target.size(0)
        metrics['dice'] = 1 - dice.data.cpu().numpy() * target.size(0)
        metrics['jaccard'] = 1 - jaccard_loss.data.cpu().numpy() * target.size(0)
    else:
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        metrics['dice_loss'] += dice.data.cpu().numpy() * target.size(0)
        metrics['jaccard_loss'] += jaccard_loss.data.cpu().numpy() * target.size(0)

    return loss


def plot_loss(train_loss, val_loss, name, out_dir):
    matplotlib.use('Agg')

    plt.plot(list(range(1, len(train_loss) + 1)), train_loss, label="Entrenamiento")
    plt.plot(list(range(1, len(val_loss) + 1)), val_loss, label="Validacion")

    plt.xlabel('Epocas')
    plt.ylabel('Loss')

    plt.title('Función {}'.format(name))
    plt.legend()

    plt.savefig(out_dir)
    plt.clf()


def train_model(name_file, model, dataset, optimizer, scheduler, dataloaders, name_model='UNet11', num_epochs=25):
    loss_history = []
    jaccard_loss_history = []
    loss_history_val = []
    jaccard_loss_history_val = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    if not os.path.exists("history"):
        os.mkdir("history")

    f = open("history/history_model_data_aug{}_{}_{}epochs.txt".format(name_file, name_model, num_epochs), "w+")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        f.write('Epoch {}/{}'.format(str(epoch), str(num_epochs - 1)) + "\n")
        print('-' * 10)
        f.write(str('-' * 10) + "\n")
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    f.write("LR" + str(param_group['lr']) + "\n")

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            itr = 0
            print("dataloader:", len(dataloaders[phase]))
            f.write("dataloader:" + str(len(dataloaders[phase])) + "\n")
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics, dataset)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, f, phase, epoch_samples)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                loss_history.append(epoch_loss)
                jaccard_loss_history.append(metrics['jaccard_loss'] / epoch_samples)
            elif phase == 'val':
                loss_history_val.append(epoch_loss)
                jaccard_loss_history_val.append(metrics['jaccard_loss'] / epoch_samples)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                f.write("saving best model" + "\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")
    print('Best val loss: {:4f}'.format(best_loss))
    f.write('Best val loss: {:4f}'.format(best_loss) + "\n")
    f.close()

    plot_loss(loss_history,
              loss_history_val,
              "Loss",
              "history/history_model_data_aug{}_{}_{}epochs_loss_chart.png".format(name_file, name_model, num_epochs))

    plot_loss(jaccard_loss_history,
              jaccard_loss_history_val,
              "Jaccard Loss",
              "history/history_model_data_aug{}_{}_{}epochs_jaccard_loss_chart.png".format(name_file, name_model, num_epochs))

    model.load_state_dict(best_model_wts)
    return model