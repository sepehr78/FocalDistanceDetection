import copy
import itertools
import multiprocessing
import os.path

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from diffractio.utils_optics import field_parameters
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


def get_phase(scalar_mask):
    amplitude, intensity, phase = field_parameters(scalar_mask.u,
                                                   has_amplitude_sign=True)
    phase[phase == 1] = -1
    phase = phase / degrees
    from diffractio.scalar_fields_XY import percentaje_intensity
    phase[intensity < percentaje_intensity * (intensity.max())] = 0
    return phase


# saved_img_size = (48, 48)
num_samples = 1100
# save_dir = "synth_images_11_mm_high_corr"
# num_imgs_per_noise = 200
# num_procs_to_use = multiprocessing.cpu_count()
# num_procs_to_use = 3  # 12 processes was too much because each one uses more than 3 GB of RAM

x0 = np.linspace(-250 * um, 250 * um, num_samples)
y0 = np.linspace(-250 * um, 250 * um, num_samples)

focal = 11 * mm
focal2 = 150 * mm
diameter = 5.5 * mm
dia2 = 1 * mm
d = 300 * mm
raylen = 150 * um
rayleigh_arr = np.arange(-raylen, raylen + 0.00001, raylen)  # Modify the stepsize by this
wavelength = 0.976 * um

u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
u0.gauss_beam(A=1, r0=(0, 0), z0=0 * mm, w0=6.85 * um, theta=0 * degrees)

# initialize lenses

t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t0.lens(r0=(0.0, 0.0), radius=diameter / 2, focal=focal, mask=True)
t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t1.lens(r0=(0.0, 0.0), radius=dia2 / 2, focal=focal2, mask=True)
t3 = Scalar_mask_XY(x0, y0, wavelength)
corr_len_arr = [50, 125, 200, 275]
std_arr = [0.05, 0.125, 0.200, 0.275]

i = 0
os.makedirs("roughness_sim",exist_ok=True)
for std in std_arr:
    for corr_len in corr_len_arr:
        t3.roughness(t=(corr_len, corr_len), s=std)
        phase = get_phase(t3)
        phase.astype(np.float32).tofile(f"roughness_sim/phase_{i}.dat")
        i += 1

        t3.draw(kind='phase', has_colorbar='vertical', colormap_kind='rainbow')
# graph = np.imag(t3.u)
# plt.imshow(graph, cmap="jet")

plt.show()
