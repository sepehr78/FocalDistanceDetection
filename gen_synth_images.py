import copy
import itertools
import multiprocessing
import os.path

import diffractio
import png
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import rcParams
from PIL import Image
from tqdm import tqdm

rcParams['figure.figsize'] = (7, 5)
rcParams['figure.dpi'] = 125
from datetime import datetime

from diffractio import degrees, mm, plt, sp, um, nm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.utils_optics import beam_width_1D, FWHM1D

"""The library bugs with colab, so we need to change some codes in the init.py

First change freq_max = psutil.cpu_freq()[2] into freq_max = psutil.cpu_freq() in line 44 in the __init__.py

Then close the, print("max frequency       : {:1.0f} GHz".format(freq_max)) in line 54 with #
"""

saved_img_size = (48, 48)
num_samples = 1000
save_dir = "synth_images"
num_imgs_per_noise = 100
num_procs_to_use = multiprocessing.cpu_count()

x0 = np.linspace(-300 * um, 300 * um, num_samples)
y0 = np.linspace(-300 * um, 300 * um, num_samples)

focal = 8 * mm
focal2 = 90 * mm
diameter = 2 * mm
dia2 = 2 * mm
d = 444 * mm
raylen = 20 * um
rayleigh_arr = np.arange(-20, 20 + 0.00001, raylen / 5)  # Modify the stepsize by this
wavelength = 0.976 * um

internal_noise_stds = np.arange(6) * 1000 * nm
external_noise_stds = np.linspace(0, 0.1, 6)


def save_noisy_images(rayleigh_idx):
    rayleigh = rayleigh_arr[rayleigh_idx]
    og_u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    og_u0.gauss_beam(A=1, r0=(0, 0), z0=0 * mm, w0=5 * um, theta=0 * degrees)

    # initialize lenses
    t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t0.lens(r0=(0.0, 0.0), radius=diameter / 2, focal=focal, mask=True)
    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.lens(r0=(0.0, 0.0), radius=dia2 / 2, focal=focal2, mask=True)

    for l in tqdm(range(num_imgs_per_noise)):
        for j, internal_noise_std in enumerate(internal_noise_stds):
            u0 = copy.deepcopy(og_u0)
            noise = 1j * np.random.normal(0, internal_noise_std, (num_samples, num_samples))
            u0.u = u0.u + noise

            z0 = focal + rayleigh  # Initial wave moving towards the first lens
            u0 = u0.RS(z=z0, verbose=False, new_field=True)

            u1 = u0 * t0  # After first lens

            u2 = u1.RS(z=d, verbose=False, new_field=True)  # Moving d distance

            u4 = u2 * t1  # After second lens
            u5 = u4.RS(z=focal2, verbose=False, new_field=True)  # After being focused on the camera

            intensity = np.abs(u5.u) ** 2  # Here is where we take the abs square of the electric field (u5.u) in

            for k, external_noise_std in enumerate(external_noise_stds):
                ext_noise = np.random.normal(0, external_noise_std, (num_samples, num_samples))

                i_w_ext_noise = (intensity - intensity.min()) / (intensity.max() - intensity.min()) + ext_noise
                img_data = ((i_w_ext_noise - i_w_ext_noise.min()) / (
                        i_w_ext_noise.max() - i_w_ext_noise.min()) * 255).astype(np.uint8)
                img = Image.fromarray(img_data, "L")
                img = img.resize(saved_img_size)
                img.save(os.path.join(save_dir, f"{rayleigh_idx}_{j}_{k}_imgs", f"{l:05d}.png"))


if __name__ == '__main__':
    # start=datetime.now()
    # Generates the noise levels ranging from 0nm to 500nm
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "info.txt"), 'w') as f:
        f.write(
            f"Focus distances: {rayleigh_arr}\nInternal noises: {internal_noise_stds}\nExternal noises: {external_noise_stds}")

    print("Creating folder structure...")
    for i, j, k in itertools.product(range(len(rayleigh_arr)), range(len(internal_noise_stds)),
                                     range(len(external_noise_stds))):
        img_dir = os.path.join(save_dir, f"{i}_{j}_{k}_imgs")
        os.makedirs(img_dir, exist_ok=True)

    total_num_imgs = len(rayleigh_arr) * len(internal_noise_stds) * len(external_noise_stds) * num_imgs_per_noise
    n_jobs = min(len(rayleigh_arr), num_procs_to_use)
    print(f"Generating {total_num_imgs} images using {n_jobs} parallel processes...")
    # get_rayleigh = lambda i: rayleigh_arr[i]
    # get_internal_noise_std = lambda i: internal_noise_stds[i]
    # get_external_noise_stds = lambda i: external_noise_stds[i]
    # get_img_save_path = lambda i, j, k, l: os.path.join(save_dir, f"{i}_{j}_{k}_imgs", f"{l:05d}.png")

    Parallel(n_jobs=n_jobs)(delayed(save_noisy_images)(i) for i in tqdm(range(len(rayleigh_arr))))
