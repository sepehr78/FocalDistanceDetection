import os
from functools import partial
import numpy as np
import torch.cuda
import torch.nn.functional as F
from fastai.callback.schedule import Learner  # To get `fit_one_cycle`, `lr_find`
from fastai.callback.tracker import SaveModelCallback
from fastai.metrics import accuracy
from fastai.optimizer import OptimWrapper
from torch import optim
from torch.utils.data import ConcatDataset, random_split

from VideoDataset import VideoDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_all_datasets(dataset_dir, predictor=False):
    video_files = os.listdir(dataset_dir)
    label_names = np.asarray([int(video_file[:video_file.find(".")]) for video_file in video_files])
    if predictor:
        min_label, max_label = np.min(label_names), np.max(label_names)

    sorted_indices = np.argsort(label_names)
    label_names = label_names[sorted_indices]
    video_files = np.asarray(video_files)[sorted_indices]

    datasets = []
    for video_file in video_files:
        label = np.where(label_names == int(video_file[:video_file.find(".")]))[0].item()
        label = np.asarray([(label_names[label] - min_label) / (max_label - min_label)],
                           dtype=np.float32) if predictor else label
        dataset = VideoDataset(os.path.join(dataset_dir, video_file), label)
        datasets.append(dataset)

    return label_names, datasets


def split_train_val(datasets, label_names, val_percentage, test_percentage, training_labels=None, seed=None):
    gen = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    training_dsets = []
    other_dsets = []
    training_labels = label_names if training_labels is None else training_labels
    for i, label in enumerate(label_names):
        if label in training_labels:
            training_dsets.append(datasets[i])
        else:
            other_dsets.append(datasets[i])

    training_dsets = ConcatDataset(training_dsets)
    val_number = int(val_percentage * len(training_dsets))
    test_number = int(test_percentage * len(training_dsets))
    train_number = len(training_dsets) - val_number - test_number
    train_dset, val_dset, test_dset = random_split(training_dsets, [train_number, val_number, test_number], gen)

    # split other datasets into val and test
    if len(other_dsets) > 0:
        other_dsets = ConcatDataset(other_dsets)
        valtest_percentage = val_percentage / (val_percentage + test_percentage)
        val_number = int(valtest_percentage * len(other_dsets))
        test_number = len(other_dsets) - val_number
        val_dset_extra, test_dset_extra = random_split(other_dsets, [val_number, test_number], gen)
        val_dset = ConcatDataset([val_dset, val_dset_extra])
        test_dset = ConcatDataset([test_dset, test_dset_extra])

    return train_dset, val_dset, test_dset


def compute_predicted_acc(predicted, target, label_names, return_preds_target=False):
    min_label, max_label = torch.min(label_names), torch.max(label_names)
    predicted = predicted * (max_label - min_label) + min_label
    pred_indices = torch.abs(predicted - label_names).argmin(dim=1)
    pred_labels = label_names[pred_indices].unsqueeze(1)  # shape (batchsize, 1)

    target = torch.round(target * (max_label - min_label) + min_label).long()

    # the class with the highest energy is what we choose as prediction
    total = pred_labels.size(0)
    correct = (pred_labels == target).sum().item()

    if return_preds_target:
        return correct / total, pred_labels, target
    return correct / total


def train_predictor(net, dls, label_names, num_epochs, path=None):
    predict_acc = lambda pred, target: compute_predicted_acc(pred, target, label_names)
    net = net.to(device)
    opt_func = partial(OptimWrapper, opt=optim.Adam)
    learn = Learner(dls, net, loss_func=F.mse_loss, opt_func=opt_func, metrics=predict_acc, path=path)
    lr = learn.lr_find(num_it=2000)[0]
    learn.fit_one_cycle(num_epochs, lr, cbs=SaveModelCallback(monitor='<lambda>', min_delta=0.001))
    learn.save("trainedModel", with_opt=False)
    return learn


def train_classifier(net, dls, num_epochs, path=None):
    net = net.to(device)
    opt_func = partial(OptimWrapper, opt=optim.Adam)
    learn = Learner(dls, net, loss_func=F.cross_entropy, opt_func=opt_func, metrics=accuracy, path=path)
    lr = learn.lr_find(num_it=2000)[0]
    learn.fit_one_cycle(num_epochs, lr, cbs=SaveModelCallback(monitor="accuracy", min_delta=0.001))
    learn.save("trainedModel", with_opt=False)
    return learn


def predict_labels_net(net, d_loader, label_names, is_predictor_model):
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in d_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            if is_predictor_model:
                _, preds, tar = compute_predicted_acc(outputs, labels, label_names, True)
            else:
                preds = label_names[outputs.argmax(dim=1)]
                tar = label_names[labels]
            predictions.extend(preds.cpu().numpy().flatten().tolist())
            targets.extend(tar.cpu().numpy().flatten().tolist())

    return np.asarray(predictions), np.asarray(targets)
