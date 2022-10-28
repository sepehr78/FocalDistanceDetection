import argparse
import pickle

from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader

from models.model import FocusPredictor, FocusClassifier
from utils.utils import *

directories = [os.path.join("training_videos", x) for x in ["Copper", "Steel", "Silicon"]]
parser = argparse.ArgumentParser(description='Train CNN to predict images on different sets of videos')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--val_p', type=float, default=0.15, help='Percentage of frames to use for validation')
parser.add_argument('--seed', type=int, default=423132, help='Seed used for splitting frames into train/val/test')
parser.add_argument('--directories', default=directories, nargs='+',
                    help='Directories where video files are located. Each '
                         'directory should contain videos at different focal '
                         'distances and the name of each video file should be '
                         'its focal distance (e.g., 150.avi)')
parser.add_argument('--results_dir', default="results_predictor_all", help='Directory where results will be saved')
parser.add_argument('--train_classifier', action='store_true')
parser.set_defaults(train_classifier=False)

parser.add_argument('--train_one_model', action='store_true')
parser.set_defaults(train_one_model=True)

parser.add_argument('--train_on_all_labels', action='store_true')
parser.set_defaults(train_on_all_labels=True)

args = parser.parse_args()

val_percentage = args.val_p
test_percentage = 0  # test on separate data
seed = args.seed
batch_size = args.bs
num_epochs = args.num_epochs

use_predictor = not args.train_classifier
train_one_model = args.train_one_model


def get_combined_dataset(directories):
    datasets_list = []
    label_names_list = []
    for directory in directories:
        result_dir = os.path.join(args.results_dir, os.path.split(directory)[-1])
        os.makedirs(result_dir, exist_ok=True)

        print(f"Loading videos in {directory}...")
        label_names_np, datasets = load_all_datasets(directory, predictor=use_predictor)
        label_names_list.append(label_names_np)
        datasets_list.append(datasets)

    assert np.all([np.array_equal(label_names_list[0], arr) for arr in label_names_list]), "Dataset labels are not " \
                                                                                           "the same! "

    combined_datasets = []
    for i in range(len(datasets_list[0])):
        combined_dataset = ConcatDataset([dataset[i] for dataset in datasets_list])
        combined_datasets.append(combined_dataset)
    return label_names_list[0], combined_datasets


def train_model(label_names_np, datasets, result_dir):
    # only train on every other label including beginning and end
    if use_predictor and not args.train_on_all_labels:
        train_label_indices = [0] + list(range(2, len(label_names_np) - 1, 2)) + [-1]
        training_labels = label_names_np[train_label_indices]
        training_labels = np.sort(list(training_labels) + [0])
    else:
        training_labels = label_names_np
    print(f"Training using videos {training_labels}...")

    train_dset, val_dset, _ = split_train_val(datasets, label_names_np, val_percentage, test_percentage,
                                              training_labels, seed)
    label_names = torch.from_numpy(label_names_np).long().to(device)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    dls = DataLoaders(train_loader, val_loader)

    if use_predictor:
        net = FocusPredictor()
        train_predictor(net, dls, label_names, num_epochs, result_dir)
    else:
        net = FocusClassifier(len(label_names_np))
        train_classifier(net, dls, num_epochs, result_dir)


if __name__ == '__main__':
    pred_str = "predictor(s)" if use_predictor else "classifier(s)"
    model_train_str = "one CNN" if train_one_model else "different CNN"

    print(f"Training {model_train_str} {pred_str} for videos in {args.directories}")
    if train_one_model:
        label_names_np, datasets = get_combined_dataset(args.directories)
        max_label = np.max(label_names_np)
        min_label = np.min(label_names_np)
        result_dir = os.path.join(args.results_dir, "combined_model")
        train_model(label_names_np, datasets, result_dir)

        pickle.dump(label_names_np, open(os.path.join(result_dir, "label_names.pkl"), "wb"))

    else:
        for directory in args.directories:
            result_dir = os.path.join(args.results_dir, os.path.split(directory)[-1])
            os.makedirs(result_dir, exist_ok=True)

            print(f"Loading videos in {directory}...")
            label_names_np, datasets = load_all_datasets(directory, predictor=use_predictor)
            max_label = np.max(label_names_np)
            min_label = np.min(label_names_np)

            train_model(label_names_np, datasets, result_dir)

            pickle.dump(label_names_np, open(os.path.join(result_dir, "label_names.pkl"), "wb"))

        # test on testing data
        # test_preds, test_targets = predict_labels(net, test_loader, label_names, use_predictor)
        # pickle.dump((test_preds, test_targets, label_names_np), open(os.path.join(result_dir, "test_pred_targets.pkl"), "wb"))
