import os
import numpy as np
import onnx
import torch.cuda
import torchvision.io
from onnx2pytorch import ConvertModel
from torch import nn, optim
from torchinfo import summary
from torch.utils.data import *
from tqdm import tqdm
from VideoDataset import VideoDataset
from model import FocusClassifier, FocusPredictor
import onnxruntime as ort

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
val_percentage = 0.15
test_percentage = 0.10
seed = 242344
batch_size = 16

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
        label = (label_names[label] - min_label) / (max_label - min_label) if predictor else label
        dataset = VideoDataset(os.path.join(dataset_dir, video_file), label)
        datasets.append(dataset)

    return label_names, datasets


def split_train_val(datasets, label_names, training_labels=None, seed=None):
    gen = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()

    # for training, only keep odd distances (-15, -13, ..., 13, 15)
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

def get_onnx_session():
    # onnx_model = onnx.load("predictorNet.onnx")
    # onnx.checker.check_model(onnx_model)
    #
    # inputs = onnx_model.graph.input
    # name_to_input = {}
    # for input in inputs:
    #     name_to_input[input.name] = input
    #
    # for initializer in onnx_model.graph.initializer:
    #     if initializer.name in name_to_input:
    #         inputs.remove(name_to_input[initializer.name])
    #
    # onnx.save(onnx_model, "netPredictorV2.onnx")

    ort_sess = ort.InferenceSession('predictorNetV2.onnx', providers=["CUDAExecutionProvider"])

    return ort_sess

def test_mathematica_model():
    ort_sess = get_onnx_session()

    tensor, _, _ = torchvision.io.read_video("videos/02_03_videos/Steel (10 micron)/0.avi")
    tensor = tensor[:, :, :, 0].unsqueeze(1) * 1.0
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = tensor[0].unsqueeze(0).numpy()

    input_name = ort_sess.get_inputs()[0].name
    output = ort_sess.run(None, {input_name: img})
    print(output)

def classifier_train():
    label_names, datasets = load_all_datasets("videos/02_03_videos/Steel (10 micron)")
    training_labels = [label_names[i] for i in range(0, len(label_names), 2)]
    print(f"Training using {training_labels}")

    train_dset, val_dset, test_dset = split_train_val(datasets, label_names, label_names[0:1], seed)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    net = FocusClassifier(len(training_labels)).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.93, weight_decay=0.001)

    num_epochs = 10
    val_acc_arr = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        train_model_one_epoch(net, train_loader, criterion, optimizer)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(inputs)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc_arr[epoch] = correct / total
            # print(val_acc_arr[epoch])

    print(val_acc_arr)

def train_model_one_epoch(net, train_loader, criterion, optimizer):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if labels.dtype == torch.float64:
            labels = labels.float()
            labels = labels.unsqueeze(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    # test_mathematica_model()
    label_names, datasets = load_all_datasets("videos/02_03_videos/Steel (10 micron)", predictor=True)
    max_label = np.max(label_names)
    min_label = np.min(label_names)
    training_labels = [label_names[i] for i in range(0, len(label_names), 2)]
    print(f"Training using {training_labels}")

    train_dset, val_dset, test_dset = split_train_val(datasets, label_names, training_labels, seed)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=2)

    label_names = torch.from_numpy(label_names).to(device)

    ort_sess = get_onnx_session()
    total = correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            # inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = torch.from_numpy(ort_sess.run(None, {"Input": inputs.numpy()})[0]).to(device)

            # outputs = outputs * (max_label - min_label) + min_label
            pred_indices = torch.abs(outputs - label_names).argmin(dim=1)
            pred_labels = label_names[pred_indices]

            labels = labels * (max_label - min_label) + min_label

            # the class with the highest energy is what we choose as prediction
            total += pred_labels.size(0)
            correct += (pred_labels == labels).sum().item()
        acc = correct / total
    print(acc)

    criterion = nn.MSELoss()
    net = FocusPredictor().to(device)
    # optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.93, weight_decay=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)


    num_epochs = 10
    val_acc_arr = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        train_model_one_epoch(net, train_loader, criterion, optimizer)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(inputs)

                # rescale outputs
                outputs = outputs * (max_label - min_label) + min_label
                pred_indices = torch.abs(outputs - label_names).argmin(dim=1)
                pred_labels = label_names[pred_indices]

                labels = labels * (max_label - min_label) + min_label

                # the class with the highest energy is what we choose as prediction
                total += pred_labels.size(0)
                correct += (pred_labels == labels).sum().item()
            val_acc_arr[epoch] = correct / total
            # print(val_acc_arr[epoch])

    print(val_acc_arr)
    # onnx_model = onnx.load("predictorNet.onnx")
    # pytorch_model = ConvertModel(onnx_model)
    # summary(pytorch_model)
    # print("sdfds")