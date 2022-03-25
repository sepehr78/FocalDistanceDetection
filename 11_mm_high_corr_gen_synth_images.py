import copy
import itertools
import multiprocessing
import os.path

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from matplotlib import rcParams
from tqdm import tqdm

from diffractio import degrees, mm, um, nm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY

"""
This script generates images with given internal and external noise stds and resizes and saves them in the given
save directory. It uses multiple cores/threads in parallel to speed up the simulations.
"""

saved_img_size = (48, 48)
num_samples = 1100
save_dir = "synth_images_11_mm_high_corr"
num_imgs_per_noise = 200
num_procs_to_use = multiprocessing.cpu_count()
num_procs_to_use = 3  # 12 processes was too much because each one uses more than 3 GB of RAM

x0 = np.linspace(-500 * um, 500 * um, num_samples)
y0 = np.linspace(-500 * um, 500 * um, num_samples)

focal = 11 * mm
focal2 = 150 * mm
diameter = 5.5 * mm
dia2 = 1 * mm
d = 300 * mm
raylen = 150 * um
rayleigh_arr = np.arange(-raylen,  raylen + 0.00001, raylen/5)  # Modify the stepsize by this
wavelength = 0.976 * um

internal_noise_stds = np.asarray([100, 500, 750, 1000]) 
external_noise_stds = np.asarray([0, 0.01, 0.03, 0.05])

def save_noisy_images(rayleigh):
    og_u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    og_u0.gauss_beam(A=1, r0=(0, 0), z0=0 * mm, w0=6.85 * um, theta=0 * degrees)

    # initialize lenses
    t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t0.lens(r0=(0.0, 0.0), radius=diameter / 2, focal=focal, mask=True)
    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.lens(r0=(0.0, 0.0), radius=dia2 / 2, focal=focal2, mask=True)
    t3 = Scalar_mask_XY(x0, y0, wavelength)
    
    for l in tqdm(range(num_imgs_per_noise)):
        for corr_len in internal_noise_stds:
            u0 = copy.deepcopy(og_u0)
            t3.roughness(t=(corr_len , corr_len), s=250*nm)
            u0 = u0*t3

            z0 = focal + rayleigh  # Initial wave moving towards the first lens
            u0 = u0.RS(z=z0, verbose=False, new_field=True)

            u1 = u0 * t0  # After first lens

            u2 = u1.RS(z=d, verbose=False, new_field=True)  # Moving d distance

            u4 = u2 * t1  # After second lens
            u5 = u4.RS(z=focal2, verbose=False, new_field=True)  # After being focused on the camera

            intensity = np.abs(u5.u) ** 2  # Here is where we take the abs square of the electric field (u5.u) in

            for external_noise_std in external_noise_stds:
                ext_noise = np.random.normal(0, external_noise_std, (num_samples, num_samples))

                i_w_ext_noise = (intensity - intensity.min()) / (intensity.max() - intensity.min()) + ext_noise
                img_data = ((i_w_ext_noise - i_w_ext_noise.min()) / (i_w_ext_noise.max() - i_w_ext_noise.min()) * 255).astype(np.uint8)
                img = Image.fromarray(img_data, "L")
                img = img.resize(saved_img_size, PIL.Image.NEAREST)  # if proper interpolation is used then ext-noise will be removed
                # plt.imshow(img, cmap="gray")
                # plt.show()
                img.save(os.path.join(save_dir, f"{round(rayleigh,1)}_{corr_len}_{external_noise_std}_imgs", f"{l:05d}.png"))


if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "info.txt"), 'w') as f:
        f.write(
            f"Focus distances: {rayleigh_arr}\nInternal noises: {internal_noise_stds}\nExternal noises: {external_noise_stds}")

    print("Creating folder structure...")
    for rayleigh, int_noise, ext_noise in itertools.product(rayleigh_arr, internal_noise_stds, external_noise_stds):
        img_dir = os.path.join(save_dir, f"{round(rayleigh,1)}_{int_noise}_{ext_noise}_imgs")
        os.makedirs(img_dir, exist_ok=True)

    total_num_imgs = len(rayleigh_arr) * len(internal_noise_stds) * len(external_noise_stds) * num_imgs_per_noise
    n_jobs = min(len(rayleigh_arr), num_procs_to_use)
    print(f"Generating {total_num_imgs} images using {n_jobs} parallel processes...")

    Parallel(n_jobs=n_jobs)(delayed(save_noisy_images)(rayleigh) for rayleigh in tqdm(rayleigh_arr))
