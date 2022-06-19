import argparse
import pickle

from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader

from model import FocusPredictor, FocusClassifier
from utils import *

directories = [os.path.join("training_videos", x) for x in ["Copper", "Steel", "Silicon"]]
parser = argparse.ArgumentParser(description='Train CNN to predict images on different sets of videos')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--val_p', type=float, default=0.15, help='Percentage of frames to use for validation')
parser.add_argument('--seed', type=int, default=423132, help='Seed used for splitting frames into train/val/test')
parser.add_argument('--directories', default=directories, nargs='+', help='Directories where video files are located. Each '
                                                                 'directory should contain videos at different focal '
                                                                 'distances and the name of each video file should be '
                                                                 'its focal distance (e.g., 150.avi)')
parser.add_argument('--results_dir', default="results_predictor", help='Directory where results will be saved')
parser.add_argument('--train_classifier', action='store_true')
parser.set_defaults(train_classifier=False)


args = parser.parse_args()

val_percentage = args.val_p
test_percentage = 0  # test on separate data
seed = args.seed
batch_size = args.bs
num_epochs = args.num_epochs

use_predictor = not args.train_classifier

if __name__ == '__main__':
    pred_str = "predictors" if use_predictor else "classifiers"
    print(f"Training different CNN {pred_str} for videos in {args.directories}")
    for directory in args.directories:
        result_dir = os.path.join(args.results_dir, os.path.split(directory)[-1])
        os.makedirs(result_dir, exist_ok=True)

        print(f"Loading videos in {directory}...")
        label_names_np, datasets = load_all_datasets(directory, predictor=use_predictor)
        max_label = np.max(label_names_np)
        min_label = np.min(label_names_np)

        # only train on every other label including beginning and end
        if use_predictor:
            train_label_indices = [0] + list(range(2, len(label_names_np) - 1, 2)) + [-1]
            training_labels = label_names_np[train_label_indices]
        else:
            training_labels = label_names_np
        print(f"Training using videos {training_labels}...")

        train_dset, val_dset, _ = split_train_val(datasets, label_names_np, val_percentage, test_percentage, training_labels, seed)
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

        pickle.dump(label_names_np, open(os.path.join(result_dir, "label_names.pkl"), "wb"))

        # test on testing data
        # test_preds, test_targets = predict_labels(net, test_loader, label_names, use_predictor)
        # pickle.dump((test_preds, test_targets, label_names_np), open(os.path.join(result_dir, "test_pred_targets.pkl"), "wb"))
