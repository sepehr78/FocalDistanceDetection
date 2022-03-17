import argparse
import os
import pickle

import pandas as pd
import sklearn.metrics
from fastai.data.load import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils import *

from model import FocusClassifier, FocusPredictor

sns.set_theme(style="white")

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.serif": 'Times New Roman',
    # # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 16,
    "font.size": 16,
    # # Make the legend/label fonts a little smaller
    # "legend.fontsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
}
plt.rcParams.update(tex_fonts)

folder_names = ["Copper", "Steel", "Silicon"]
directories = [os.path.join("testing_videos", x) for x in folder_names]
results_directories = [os.path.join("results_predictor", x) for x in folder_names]

parser = argparse.ArgumentParser(description='Plot confusion matrices and report accuracies of trained models on test '
                                             'data. train_cnn.py must be run before')
parser.add_argument('--results_directories', nargs='+', default=results_directories,
                    help='Directory where results are saved')
parser.add_argument('--directories', default=directories, nargs='+',
                    help='Directories where video files are located for testing. Each '
                         'directory should contain videos at different focal '
                         'distances and the name of each video file should be '
                         'its focal distance (e.g., 150.avi)')
parser.add_argument('--test_classifier', action='store_true')
parser.set_defaults(test_classifier=False)

args = parser.parse_args()

use_classifier = args.test_classifier


def plot_cm_matrix(targets, predictions, label_names, file_name=None):
    cm = confusion_matrix(targets, predictions, labels=label_names, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig = plt.figure(figsize=(6.4 * 1.1, 4.8 * 1.1))
    # disp.plot(include_values=False, ax=fig.gca())
    df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
    sns.heatmap(df_cm, cmap="flare", linewidths=.5, square=True, vmin=0, vmax=1)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    results_directories = args.results_directories
    testing_directories = args.directories
    time_taken_arr = []
    for directory, result_dir in zip(testing_directories, results_directories):
        label_names_np = pickle.load(open(os.path.join(result_dir, "label_names.pkl"), "rb"))
        net = FocusClassifier(len(label_names_np)).to(device) if use_classifier else FocusPredictor().to(device)
        net.load_state_dict(torch.load(os.path.join(result_dir, "models/trainedModel.pth")))

        print(f"Loading videos in {directory}...")
        label_names_testing, datasets = load_all_datasets(directory, predictor=not use_classifier)
        assert np.all(label_names_testing == label_names_np)

        label_names = torch.from_numpy(label_names_np).long().to(device)

        testing_dl = DataLoader(ConcatDataset(datasets), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        test_preds, test_targets, avg_time_taken = predict_labels_net(net, testing_dl, label_names, not use_classifier)
        time_taken_arr.append(avg_time_taken)

        class_report_txt = classification_report(test_targets, test_preds, target_names=label_names_np.astype(str))
        print(class_report_txt)
        with open(os.path.join(result_dir, "testing_report.txt"), 'w') as f:
            f.write(class_report_txt)

        class_report_dict = classification_report(test_targets, test_preds, target_names=label_names_np.astype(str),
                                                  output_dict=True)
        with open(os.path.join(result_dir, "class_report_dict.pkl"), 'wb') as f:
            pickle.dump(class_report_dict, f)

        plot_cm_matrix(test_targets, test_preds, label_names_np, os.path.join(result_dir, "testing_cm.pdf"))

    print(f"CNN processed {1 / np.mean(time_taken_arr):.2f} images/sec when using {device}")
