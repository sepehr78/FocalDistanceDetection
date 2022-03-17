import argparse
import pickle

from fastai.data.block import DataBlock, RegressionBlock, CategoryBlock
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.data.transforms import get_image_files, RandomSplitter, IntToFloatTensor
from fastai.vision.core import PILImageBW
from fastai.vision.data import ImageBlock
from sklearn.metrics import classification_report

from model import FocusClassifier
from test_cnn import plot_cm_matrix
from utils import *

parser = argparse.ArgumentParser(description='Plot confusion matrices and report accuracies of trained models on test '
                                             'data. train_cnn_on_synth.py must be run before')
parser.add_argument('--synth_imgs_dir', default="synth_images", help='Directory where generated synthetic images for '
                                                                     'testing from gen_synth_images.py are located.')
parser.add_argument('--results_dir', default="results_synth_classifier", help='Directory where results will be saved')

args = parser.parse_args()


def get_label_from_fname(fname):
    return float(fname.parts[-2].split("_")[0])


if __name__ == '__main__':
    result_dir = args.results_dir
    os.makedirs(result_dir, exist_ok=True)

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
