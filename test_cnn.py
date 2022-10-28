import argparse
import os
import pickle

import pandas as pd
import sklearn.metrics
from fastai.data.load import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils import *

from model import FocusClassifier, FocusPredictor

folder_names = ["Copper", "Steel", "Silicon"]
directories = [os.path.join("testing_videos", x) for x in folder_names]
# results_directories = [os.path.join("results_predictor", x) for x in folder_names]

parser = argparse.ArgumentParser(description='Plot confusion matrices and report accuracies of trained models on test '
                                             'data. train_cnn.py must be run before')
# parser.add_argument('--results_directories', nargs='+', default=results_directories,
#                     help='Directory where results are saved (for different CNN models) and where the results of '
#                          'testing, like confusion matrix, will be saved')
parser.add_argument('--results_dir', default="results_predictor_all", help='Directory where results are saved')

parser.add_argument('--directories', default=directories, nargs='+',
                    help='Directories where video files are located for testing. Each '
                         'directory should contain videos at different focal '
                         'distances and the name of each video file should be '
                         'its focal distance (e.g., 150.avi)')
parser.add_argument('--test_classifier', action='store_true')
parser.set_defaults(test_classifier=False)

parser.add_argument('--save_unscaled_out', action='store_true')
parser.set_defaults(save_unscaled_out=True)

parser.add_argument('--one_model', action='store_true')
parser.set_defaults(one_model=True)

args = parser.parse_args()

use_classifier = args.test_classifier
one_model = args.one_model


def plot_cm_matrix(cm, label_names, file_name=None):
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
    results_dir = args.results_dir
    testing_directories = args.directories
    time_taken_arr = []
    if one_model:
        result_dir = os.path.join(results_dir, "combined_model")
        label_names_np = pickle.load(open(os.path.join(result_dir, "label_names.pkl"), "rb"))
        net = FocusClassifier(len(label_names_np)).to(device) if use_classifier else FocusPredictor().to(device)
        net.load_state_dict(torch.load(os.path.join(result_dir, "models/trainedModel.pth")))

    for testing_dir in testing_directories:
        result_dir = os.path.join(args.results_dir, os.path.split(testing_dir)[-1])
        os.makedirs(result_dir, exist_ok=True)
        if not one_model:
            label_names_np = pickle.load(open(os.path.join(result_dir, "label_names.pkl"), "rb"))
            net = FocusClassifier(len(label_names_np)).to(device) if use_classifier else FocusPredictor().to(device)
            net.load_state_dict(torch.load(os.path.join(result_dir, "models/trainedModel.pth")))

        print(f"Loading videos in {testing_dir}...")
        label_names_testing, datasets = load_all_datasets(testing_dir, predictor=not use_classifier)
        assert np.all(label_names_testing == label_names_np)

        label_names = torch.from_numpy(label_names_np).long().to(device)

        testing_dl = DataLoader(ConcatDataset(datasets), batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

        if args.save_unscaled_out:
            net_output, test_targets, _ = get_net_outout_and_time(net, testing_dl, label_names)
            net_output.astype(np.float32).tofile(os.path.join(result_dir, "net_out.dat"))
            test_targets.astype(int).tofile(os.path.join(result_dir, "targets.dat"))

        test_preds, test_targets, avg_time_taken = predict_labels_net(net, testing_dl, label_names, not use_classifier)
        time_taken_arr.append(avg_time_taken)

        class_report_txt = classification_report(test_targets, test_preds, target_names=label_names_np.astype(str),
                                                 digits=4)
        print(class_report_txt)
        with open(os.path.join(result_dir, "testing_report.txt"), 'w') as f:
            f.write(class_report_txt)

        class_report_dict = classification_report(test_targets, test_preds, target_names=label_names_np.astype(str),
                                                  output_dict=True)
        with open(os.path.join(result_dir, "class_report_dict.pkl"), 'wb') as f:
            pickle.dump(class_report_dict, f)

        cm = confusion_matrix(test_targets, test_preds, labels=label_names_np, normalize='true')
        cm.astype(np.float32).tofile(os.path.join(result_dir, "cm.dat"))

        plot_cm_matrix(cm, label_names_np, os.path.join(result_dir, "testing_cm.pdf"))

    print(f"CNN processed {1 / np.mean(time_taken_arr):.2f} images/sec when using {device}")
