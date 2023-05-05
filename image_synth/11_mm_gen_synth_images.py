import copy
import multiprocessing
import os.path
import sys

import PIL.Image
import numpy as np
from PIL import Image

# temporarily block printing when importing diffractio
sys.stdout = open(os.devnull, 'w')
from diffractio import degrees, mm, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
sys.stdout = sys.__stdout__

from joblib import Parallel, delayed
from tqdm import tqdm
import h5py

"""
This script generates images with given internal and external noise stds and resizes and saves them in the given
save directory. It uses multiple cores/threads in parallel to speed up the simulations.
"""

saved_img_size = (50, 50)
num_samples = 1100
save_dir = "synth_images_11_mm"
num_imgs_per_config = 200
num_procs_to_use = multiprocessing.cpu_count()

x0 = np.linspace(-500 * um, 500 * um, num_samples)
y0 = np.linspace(-500 * um, 500 * um, num_samples)

focal = 11 * mm
focal2 = 150 * mm
diameter = 5.5 * mm
dia2 = 1 * mm
d = 300 * mm
raylen = 150 * um
rayleigh_arr = np.arange(-raylen, raylen + 0.00001, raylen / 5)  # Modify the stepsize by this
wavelength = 0.976 * um

og_u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
og_u0.gauss_beam(A=1, r0=(0, 0), z0=0 * mm, w0=6.85 * um, theta=0 * degrees)

# initialize lenses
t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t0.lens(r0=(0.0, 0.0), radius=diameter / 2, focal=focal, mask=True)
t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t1.lens(r0=(0.0, 0.0), radius=dia2 / 2, focal=focal2, mask=True)
t3 = Scalar_mask_XY(x0, y0, wavelength)

# corr_len_arr = np.asarray([50, 125, 250, 500])
# internal_noise_stds = np.asarray([0, 0.25, 0.5, 0.75])
# external_noise_stds = np.asarray([0, 0.01, 0.03, 0.05])
corr_len_arr = np.linspace(50, 500, 10)
internal_noise_stds = np.linspace(0, 0.50, 10)
external_noise_stds = np.linspace(0, 0.05, 10)


def get_folder_path(rayleigh, corr_len, int_noise, ext_noise):
    return os.path.join(save_dir, f"{round(rayleigh, 1)}_{corr_len}_{int_noise}_{ext_noise}_imgs")


def gen_imgs():
    """
    Generates an ndarray containing image data after simulation for each rayleigh length, int. noise, correlation length,
    and ext. noise.
    :return: Image ndarray of size (len(rayleigh_arr), len(internal_noise_stds), len(corr_len_arr), len(external_noise_stds)) + saved_img_size
    """

    imgs_arr = np.zeros(
        (len(rayleigh_arr), len(internal_noise_stds), len(corr_len_arr), len(external_noise_stds)) + saved_img_size,
        np.uint8)
    for i, rayleigh in enumerate(rayleigh_arr):
        for j, int_noise in enumerate(internal_noise_stds):
            for k, corr_len in enumerate(corr_len_arr):
                u0 = copy.deepcopy(og_u0)
                t3.roughness(t=(corr_len, corr_len), s=int_noise)
                u0 = u0 * t3

                z0 = focal + rayleigh  # Initial wave moving towards the first lens
                u0 = u0.RS(z=z0, verbose=False, new_field=True)

                u1 = u0 * t0  # After first lens

                u2 = u1.RS(z=d, verbose=False, new_field=True)  # Moving d distance

                u4 = u2 * t1  # After second lens
                u5 = u4.RS(z=focal2, verbose=False, new_field=True)  # After being focused on the camera

                intensity = np.abs(u5.u) ** 2  # Here is where we take the abs square of the electric field (u5.u) in

                for l, ext_noise in enumerate(external_noise_stds):
                    ext_noise_arr = np.random.normal(0, ext_noise, (num_samples, num_samples))

                    i_w_ext_noise = (intensity - intensity.min()) / (intensity.max() - intensity.min()) + ext_noise_arr
                    img_data = ((i_w_ext_noise - i_w_ext_noise.min()) / (
                            i_w_ext_noise.max() - i_w_ext_noise.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(img_data, "L")
                    img_data_resized = np.asarray(img.resize(saved_img_size, PIL.Image.NEAREST))
                    imgs_arr[i, j, k, l] = img_data_resized

    return imgs_arr


if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    info_text = (f"Focus distances: {rayleigh_arr}\n"
                 f"Correlation lengths: {corr_len_arr}\n"
                 f"Internal noises: {internal_noise_stds}\n"
                 f"External noises: {external_noise_stds}")
    print(info_text)
    with open(os.path.join(save_dir, "info.txt"), 'w') as f:
        f.write(info_text)

    print("Creating HDF5 datasets for each Rayleigh length...")
    for rayleigh in tqdm(rayleigh_arr):
        f_list = []
        for rayleigh in rayleigh_arr:
            f = h5py.File(os.path.join(save_dir, f"rayleigh_{rayleigh:.0f}.hdf5"), "w")
            f_list.append(f)

            # create HDF5 dataset for each correlation, int. noise, ext. noise triplet
            for i, int_noise in enumerate(internal_noise_stds):
                for j, corr_len in enumerate(corr_len_arr):
                    for k, ext_noise in enumerate(external_noise_stds):
                        dset = f.create_dataset(f"{i}_{j}_{k}", (num_imgs_per_config,) + saved_img_size, np.uint8)
                        dset.attrs["rayleigh"] = rayleigh
                        dset.attrs["corr_len"] = corr_len
                        dset.attrs["int_noise"] = int_noise
                        dset.attrs["ext_noise"] = ext_noise

    total_num_imgs = len(rayleigh_arr) * len(internal_noise_stds) * len(corr_len_arr) * len(external_noise_stds) \
                     * num_imgs_per_config

    print(
        f"Generating {num_imgs_per_config} images per config ({total_num_imgs} images in total) using {num_procs_to_use} parallel processes...")

    img_idx_arr = np.arange(num_imgs_per_config)
    img_idx_chunks = [img_idx_arr[i:i + num_procs_to_use] for i in range(0, len(img_idx_arr), num_procs_to_use)]
    try:
        with Parallel(n_jobs=num_procs_to_use) as parallel:
            with tqdm(total=num_imgs_per_config) as pbar:
                for img_idx_chunk in img_idx_chunks:
                    # use multi-processing to generate images
                    imgs_arr_list = parallel(delayed(gen_imgs)() for _ in img_idx_chunk)

                    # write to H5 dataset
                    for imgs_arr, img_idx in zip(imgs_arr_list, img_idx_chunk):
                        for i in range(len(rayleigh_arr)):
                            for j in range(len(internal_noise_stds)):
                                for k in range(len(corr_len_arr)):
                                    for l in range(len(external_noise_stds)):
                                        f_list[i][f"{j}_{k}_{l}"][img_idx] = imgs_arr[i, j, k, l]
                    pbar.update(len(img_idx_chunk))
    finally:
        for f in f_list:
            f.close()
