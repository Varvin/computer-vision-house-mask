import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt

img = load_img('data/train/annotations/task-1-annotation-1-by-1-tag-House-0.png', target_size=(207,166))

plt.imshow(img)