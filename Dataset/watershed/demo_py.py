# Vinay Chandragiri - IITG

import PIL
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage import morphology
from skimage import segmentation
from skimage import io
import skimage

file = "/opt/TFM/DATASETS/Images_to_Inference/Dr7_587725551200895094.jpg"


def image_show(image, nrows=1, ncols=1, cmap='gray', **kwargs):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


from skimage.io import imread, imshow
from skimage import data


from skimage.io import imread, imshow
import matplotlib.pyplot as plt

image_gray = imread(file, as_gray=True)
imshow(image_gray)
plt.show()


image = data.astronaut()
imshow(image)


# WaterShed Segmentation

t = io.imread(file)
image_show(t)
plt.show()


image = image_gray
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
markers = morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)
ans = skimage.segmentation.mark_boundaries(image, labels_ws, color=(1, 1, 0), outline_color=(1,0,1), mode='inner', background_label=0)
scipy.misc.imsave('outfile.jpg', ans)

