import glob
import os.path
import numpy as np
import h5py
from tqdm import tqdm

h5_files_dir = "synth_images_11_mm"
val_split = 0.15
test_split = 0.15
seed = 542646

if __name__ == '__main__':

    rng = np.random.default_rng(seed)
    train_dir = os.path.join(h5_files_dir, "train")
    val_dir = os.path.join(h5_files_dir, "val")
    test_dir = os.path.join(h5_files_dir, "test")
    for dir in [train_dir, val_dir, test_dir]:
        os.makedirs(dir, exist_ok=True)

    for h5_path in tqdm(glob.glob(os.path.join(h5_files_dir, "*.hdf5"))):
        h5_file_name = os.path.split(h5_path)[-1]
        f = h5py.File(h5_path, "r")
        train_f = h5py.File(os.path.join(train_dir, h5_file_name), "w")
        val_f = h5py.File(os.path.join(val_dir, h5_file_name), "w")
        test_f = h5py.File(os.path.join(test_dir, h5_file_name), "w")

        for key in f:
            # calculate train,val,test indices
            num_val, num_test = round(val_split * len(f[key])), round(test_split * len(f[key]))
            num_train = len(f[key]) - num_test - num_val

            indices = set(range(len(f[key])))
            val_indices = rng.choice(len(f[key]), num_val, replace=False)
            test_indices = rng.choice(list(indices - set(val_indices)), num_test, replace=False)
            train_indices = list(indices - set(val_indices) - set(test_indices))

            # add data to train,val,test h5
            train_dset = train_f.create_dataset(key, dtype=np.uint8, data=f[key][:][train_indices])
            val_dset = val_f.create_dataset(key, dtype=np.uint8, data=f[key][:][val_indices])
            test_dset = test_f.create_dataset(key, dtype=np.uint8, data=f[key][:][test_indices])

            for dset in [train_dset, val_dset, test_dset]:
                dset.attrs.update(f[key].attrs)

        for h5file in [f, train_f, val_f, test_f]:
            h5file.close()

    print("Done")