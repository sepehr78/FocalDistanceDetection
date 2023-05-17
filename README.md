# High-Precision Laser Focus Positioning of Rough Surfaces by Deep Learning
This repository contains the code for the paper titled "High-precision laser focus positioning of rough surfaces by deep learning", published in Optics and Lasers in Engineering. For any questions regarding the code, contact [Sepehr Elahi](https://sepehrelahi.com) at [sepehr.elahi@epfl.ch](mailto:sepehr.elahi@epfl.ch).
![The CNN model used in our work](https://raw.githubusercontent.com/sepehr78/FocalDistanceDetection/master/model.png)
# Setup
The code uses standard Python numerical computing, like Pandas and NumPy, for general numerical computation, [fastai](https://docs.fast.ai) and PyTorch for training and testing CNNs, and [diffractio](https://diffractio.readthedocs.io/en/latest/) for Fourier optics simulations. We listed all the Python package requirements in the requirements.txt file, which can be installed using the following command:

    pip install -r requirements.txt
# Fourier optics simulations
The image_synth directory contains the script for simulation our setup and synthetically generating the images formed on the camera, as explained in Section 3 of the paper. Note that the script can take a very long time to run, and to speed it up we recommend running it on a powerful multi-core computer and setting the `num_procs_to_use` variable appropriately (by default it equals the number of CPU cores).

## Generating images from scratch
To generate the images used to train the synthetic CNN, simply run

    python image_synth/11_mm_gen_synth_images.py
The script will create an HDF5 dataset containing the images. 

## Downloading generated images
We provide images that we generated and used for training, validating, and testing the synthetic CNN (explained in Section 4.2) to download below: 

> Generated synthetic images download link: https://1drv.ms/f/s!Av7-zYTFLro0r5oDdE71fVy-O37P_Q?e=EtZwYs

# Experimental data
You can download the video files that we used to train our model and derive our results below:

> Training videos download link: https://1drv.ms/f/s!Av7-zYTFLro0r4ZhTm1KvrdLJ1oTew?e=8dpJUa

# Training models
You can find the scripts for training our model on the synthetic and experimental data in the training_scripts directory. We also provided a train_scikit.py script that allows for training [scikit-learn](https://scikit-learn.org/stable/) models on the data.

# Testing models
You can find the scripts for testing our model after it has been trained on the synthetic and experimental data in the testing_scripts directory.

![Laser defocus prediction results of our CNN model on different surfaces](https://github.com/sepehr78/FocalDistanceDetection/blob/master/results.png?raw=true)