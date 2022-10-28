import argparse
import pickle
from glob import glob

import pandas as pd
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import get_image_files, RandomSplitter, IntToFloatTensor
from fastai.vision.core import PILImageBW
from fastai.vision.data import ImageBlock
from sklearn.metrics import *

from models.model import FocusClassifier
from test_cnn import plot_cm_matrix
from utils.utils import *

parser = argparse.ArgumentParser(description='Plot confusion matrices and report accuracies of trained models on test '
                                             'data. train_cnn_on_synth.py must be run before')
parser.add_argument('--synth_imgs_dir', default="synth_images_testing",
                    help='Directory where generated synthetic images for '
                         'testing from gen_synth_images.py are located.')
parser.add_argument('--results_dir', default="results_synth_classifier", help='Directory where results will be saved')

args = parser.parse_args()


def get_label_from_fname(fname):
    return float(fname.parts[-2].split("_")[0])


def get_df_from_synth_img_dir(img_dir):
    """
    Returns a pandas DataFrame with containing focus and noise info of each img folder
    """
    folder_paths = glob(os.path.join(img_dir, "*/"), recursive=False)
    row_list = []
    for folder_path in folder_paths:
        folder_name = os.path.split(folder_path)[0].split("/")[-1]
        row = [float(x) for x in folder_name.split("_")[:-1]] + [folder_name]
        row_list.append(row)

    return pd.DataFrame(row_list, columns=["focus", "int_noise", "ext_noise", "path"])


def plot_noise_acc(acc_mat, int_noise_arr, ext_noise_arr, file_name):
    fig = plt.figure(figsize=(6.4 * 1.1, 4.8 * 1.1))
    # disp.plot(include_values=False, ax=fig.gca())
    int_noise_arr = np.round(int_noise_arr * 100) / 100.0  # hacky way to format .2f
    ext_noise_arr = np.round(ext_noise_arr * 100) / 100.0
    df_cm = pd.DataFrame(acc_mat, index=int_noise_arr, columns=ext_noise_arr)
    sns.heatmap(df_cm, cmap="flare", linewidths=.5, square=True, vmin=0, vmax=1, annot=True, fmt=".2f")
    plt.xlabel("Output noise std.")
    plt.ylabel("Input noise std.")
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    result_dir = args.results_dir
    os.makedirs(result_dir, exist_ok=True)

    df = get_df_from_synth_img_dir(args.synth_imgs_dir)

    int_noise_arr = np.unique(df["int_noise"].values)
    ext_noise_arr = np.unique(df["ext_noise"].values)

    num_classes = len(np.unique(df["focus"].values))

    net = FocusClassifier(num_classes, img_size=(48, 48)).to(device)
    net.load_state_dict(torch.load(os.path.join(result_dir, "models/trainedModel.pth")))

    acc_mat = np.zeros((len(int_noise_arr), len(ext_noise_arr)))
    for i, int_noise in enumerate(int_noise_arr):
        for j, ext_noise in enumerate(ext_noise_arr):
            folders = list(
                df[np.logical_and(df["int_noise"] == int_noise, df["ext_noise"] == ext_noise)]["path"].values)
            get_items_fun = lambda path: get_image_files(path, folders=folders)
            dblock = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                               get_items=get_items_fun,
                               get_y=get_label_from_fname,
                               item_tfms=IntToFloatTensor(),
                               splitter=RandomSplitter(valid_pct=0))
            test_loader = dblock.dataloaders(args.synth_imgs_dir, batch_size=64, shuffle=True, num_workers=4,
                                             pin_memory=True, device=device)[0]
            label_names = torch.tensor(test_loader.vocab.items).to(device)
            label_names_np = label_names.cpu().numpy().astype(int)
            test_preds, test_targets, avg_time_taken = predict_labels_net(net, test_loader, label_names, False)
            acc = accuracy_score(test_targets, test_preds)
            acc_mat[i, j] = acc

    plot_noise_acc(acc_mat, int_noise_arr, ext_noise_arr, os.path.join(result_dir, "noise_acc.pdf"))


    dblock = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                       get_items=get_image_files,
                       get_y=get_label_from_fname,
                       item_tfms=IntToFloatTensor(),
                       splitter=RandomSplitter(valid_pct=0))
    test_loader = dblock.dataloaders(args.synth_imgs_dir, batch_size=256, shuffle=True, num_workers=4, pin_memory=True,
                                     device=device)[0]
    label_names = torch.tensor(test_loader.vocab.items).to(device)
    label_names_np = label_names.cpu().numpy().astype(int)
    print(f"Loaded focal distances (classes) are {label_names_np}")
    temp_batch = next(iter(test_loader))
    net = FocusClassifier(len(test_loader.vocab), img_size=temp_batch[0].size()[-2:]).to(device)
    net.load_state_dict(torch.load(os.path.join(result_dir, "models/trainedModel.pth")))

    test_preds, test_targets, avg_time_taken = predict_labels_net(net, test_loader, label_names, False)
    class_report_txt = classification_report(test_targets, test_preds, target_names=label_names_np.astype(str))
    print(class_report_txt)
    with open(os.path.join(result_dir, "testing_report.txt"), 'w') as f:
        f.write(class_report_txt)

    class_report_dict = classification_report(test_targets, test_preds, target_names=label_names_np.astype(str),
                                              output_dict=True)
    with open(os.path.join(result_dir, "class_report_dict.pkl"), 'wb') as f:
        pickle.dump(class_report_dict, f)

    plot_cm_matrix(test_targets, test_preds, label_names_np, os.path.join(result_dir, "testing_cm.pdf"))
    plt.show()
