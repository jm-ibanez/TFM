import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import argparse as args

import sys

# from: https://felipemeganha.medium.com/image-segmentation-with-watershed-using-python-f40f2f7e9f40


if sys.argv[1]:
    file = sys.argv[1]
else:
    file = "/opt/TFM/DATASETS/Images_to_Inference/Dr7_587725551200895094.jpg"

# Read image

img = cv2.imread(file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


contornos, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
buracos = []
for con in contornos:
  area = cv2.contourArea(con)
  if area < 100:
    buracos.append(con)
cv2.drawContours(thresh, buracos, -1, 255, -1)


dist = ndi.distance_transform_edt(thresh)
dist_visual = dist.copy()


local_max = peak_local_max(dist, indices=False, min_distance=20, labels=thresh)

markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]

# Apply Watershed
labels = watershed(-dist, markers, mask=thresh)

# Save segmentation file
new_file = file.replace(".jpg", "_pm.jpg")
print(new_file)
l2 = np.array(labels * 255, dtype = np.uint8)
#cv2.applyColorMap(labels, cv2.COLORMAP_JET)
cv2.imwrite(new_file, cv2.applyColorMap(l2, cv2.COLORMAP_JET))


# Show segmentation results 

titulos = ['Original image', 'Binary Image', 'Distance Transform', 'Watershed']
imagens = [img, thresh, dist_visual, labels]
fig = plt.gcf()
fig.set_size_inches(16, 12)  
for i in range(4):
    plt.subplot(2,2,i+1)
    if (i == 3):
      cmap = "jet"
    else:
       cmap = "gray"
    plt.imshow(imagens[i], cmap)
    plt.title(titulos[i]) 
    plt.xticks([]),plt.yticks([])     

plt.show()


# =====
# Otsu
# =====
import cv2 as cv

# global thresholding
img = gray
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

# 



