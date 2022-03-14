import argparse

from fastai.data.block import DataBlock, RegressionBlock, CategoryBlock
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.data.transforms import get_image_files, RandomSplitter, IntToFloatTensor
from fastai.vision.core import PILImageBW
from fastai.vision.data import ImageBlock

from model import FocusClassifier
from utils import *

parser = argparse.ArgumentParser(description='Train regression CNN to predict simulated images')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--val_p', type=float, default=0.15, help='Percentage of frames to use for validation')
parser.add_argument('--seed', type=int, default=423132, help='Seed used for splitting frames into train/val/test')
parser.add_argument('--synth_imgs_dir', default="synth_images", help='Directory where generated synthetic images from '
                                                                     'gen_synth_images.py are located.')
parser.add_argument('--results_dir', default="results_synth_predictor", help='Directory where results will be saved')

args = parser.parse_args()

val_percentage = args.val_p
seed = args.seed
batch_size = args.bs
num_epochs = args.num_epochs


def get_label_from_fname(fname):
    return float(fname.parts[-2].split("_")[0])


if __name__ == '__main__':
    dblock = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                       get_items=get_image_files,
                       get_y=get_label_from_fname,
                       splitter=RandomSplitter(val_percentage, seed=seed),
                       item_tfms=IntToFloatTensor())
    dls = dblock.dataloaders(args.synth_imgs_dir, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, device=device)
    print(f"Loaded focal distances (classes) are {dls.vocab}")

    temp_batch = next(iter(dls[0]))
    net = FocusClassifier(len(dls.vocab), img_size=temp_batch[0].size()[-2:]).to(device)
    result_dir = args.results_dir
    os.makedirs(result_dir, exist_ok=True)

    train_classifier(net, dls, num_epochs, result_dir)
