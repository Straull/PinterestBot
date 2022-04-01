import numpy
from sklearn.cluster import KMeans
import numpy as np
import shutil
import matplotlib.colors as colors
from PIL import Image
from collections import Counter
from skimage.color import rgb2lab, deltaE_ciede2000
import os


def get_image(image_name):
    image = Image.open(image_name)
    image = numpy.array(image)
    return image


def main_colors(image_name, number_of_colors):
    image = get_image(image_name)
    image = image.reshape(image.shape[0] * image.shape[1], 3)

    cluster = KMeans(n_clusters=number_of_colors)
    labels = cluster.fit_predict(image)

    counts = Counter(labels)
    colors = cluster.cluster_centers_

    ordered_colors = [colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return rgb_colors


def matching_color(image_name, color, image_path, max_diff=25, numbers_of_colors=9):
    colors_of_image = main_colors(image_name, numbers_of_colors)
    color_1 = False
    color_2 = False
    color_3 = False
    color = [colors.hex2color(color[i]) for i in range(len(color))]
    wanted_colors = [rgb2lab(np.uint8(np.asarray([[color[i]]]))) for i in range(len(color))]
    for i in range(numbers_of_colors):
        current = rgb2lab(np.uint8(np.asarray([[colors_of_image[i]]])))
        if max_diff >= deltaE_ciede2000(current, wanted_colors[0]):
            color_1 = True
        if max_diff >= deltaE_ciede2000(current, wanted_colors[1]):
            color_2 = True
        if max_diff >= deltaE_ciede2000(current, wanted_colors[2]):
            color_3 = True
    if color_1 and color_2 and color_3:
        shutil.copyfile(image_name, image_path + image_name)
    os.remove(image_name)
    return
