import argparse
import pickle

import sklearn.svm
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model import FocusPredictor, FocusClassifier
from utils import *

directories = [os.path.join("training_videos", x) for x in ["Copper", "Steel", "Silicon"]]
parser = argparse.ArgumentParser(description='Train CNN to predict images on different sets of videos')
parser.add_argument('--val_p', type=float, default=0.15, help='Percentage of frames to use for validation')
parser.add_argument('--seed', type=int, default=423132, help='Seed used for splitting frames into train/val/test')
parser.add_argument('--pca_var_explained', type=float, default=0.90, help='Percentage of variance to be explained by PCA components.')

parser.add_argument('--directories', default=directories, nargs='+',
                    help='Directories where video files are located. Each '
                         'directory should contain videos at different focal '
                         'distances and the name of each video file should be '
                         'its focal distance (e.g., 150.avi)')
parser.add_argument('--results_dir', default="results_scikit", help='Directory where results will be saved')

models = [LogisticRegression(max_iter=1000), SVC()]
model_names = ["lr", "svc"]

args = parser.parse_args()

val_percentage = args.val_p
test_percentage = 0  # test on separate data
seed = args.seed

if __name__ == '__main__':
    print(f"Training different {model_names} classifiers for videos in {args.directories}")
    for directory in args.directories:
        result_dir = os.path.join(args.results_dir, os.path.split(directory)[-1])
        trained_models_dir = os.path.join(result_dir, "models")
        os.makedirs(trained_models_dir, exist_ok=True)

        print(f"Loading videos in {directory}...")
        label_names, datasets = load_all_datasets(directory, predictor=False)
        max_label = np.max(label_names)
        min_label = np.min(label_names)

        train_dset, val_dset, _ = split_train_val(datasets, label_names, val_percentage, test_percentage,
                                                  label_names, seed)

        for model, model_name in zip(models, model_names):
            print(f"Training {model}...")
            pca = PCA(n_components=args.pca_var_explained)
            scaler = StandardScaler()
            pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("model", model)], memory="scikit_cache")

            x_arr = np.asarray([x[0].numpy() for x in train_dset]).reshape(len(train_dset), -1)
            y_arr = label_names[np.asarray([x[1] for x in train_dset])]
            pipeline.fit(x_arr, y_arr)
            print(f"Using {pipeline['pca'].n_components_} for pca")
            pickle.dump(pipeline, open(os.path.join(trained_models_dir, f"{model_name}.pkl"), "wb"))

        pickle.dump(label_names, open(os.path.join(result_dir, "label_names.pkl"), "wb"))
