# please first install imageio by 'conda install -c conda-forge imageio' or 'pip install imageio'
import imageio.v2 as iio
import numpy as np

# your image paths
path = ""

# the variable suffix of the images
# example: for epoch_0.png, epoch_1.png, epoch_2.png, the suffix is '1', '2', '3' ...
variable_names = []
png_names = [path + name + ".png" for name in variable_names]
frames = []
for name in png_names:
    frames.append(iio.imread(name))

# save the gif
gif_path = path + 'restoration_effect.gif'
# duration for each image in ms
duration = 400
iio.mimsave(gif_path, frames, format="GIF", duration=duration)
