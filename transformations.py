from PIL import Image
import numpy as np
from plantcv import plantcv as pcv

fillcolor = "#fff"
scalingFactor = 0.15
BLUE = [0, 0, 255]
GREEN = [0, 255, 0]
RED = [255, 0, 0]


def get_roi_rectangle(img_mask: np.ndarray):
    nonzero_coords = np.column_stack(np.where(img_mask > 0))
    y_min, x_min = nonzero_coords.min(axis=0)
    y_max, x_max = nonzero_coords.max(axis=0)
    return y_min, y_max, x_min, x_max


def gray_scale(img: Image.Image):
    np_img = np.asarray(img)
    new_img = pcv.rgb2gray_lab(rgb_img=np_img, channel="a")
    new_img = pcv.threshold.otsu(new_img, "dark")
    return new_img


def draw_point_on_image(image, coords, color, radius=4):
    for coord in coords:
        x, y = coord[0]
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i**2 + j**2 <= radius**2 and \
                    0 <= x+i < image.shape[1] and \
                        0 <= y+j < image.shape[0]:
                    image[y+j, x+i] = color


def pseudolandmarks(img: Image.Image):
    np_img = np.asarray(img)
    img_mask = gray_scale(img)
    top, bottom, center = pcv.homology.x_axis_pseudolandmarks(np_img, img_mask)

    new_img = np_img.copy()

    draw_point_on_image(new_img, top.astype(int), RED)
    draw_point_on_image(new_img, bottom.astype(int), BLUE)
    draw_point_on_image(new_img, center.astype(int), GREEN)
    return new_img


def gaussian_blur(img: Image.Image):
    np_img = gray_scale(img)
    return pcv.gaussian_blur(np_img, (3, 3))


def mask(img: Image.Image):
    np_img = np.asarray(img)
    img_mask = gray_scale(img)
    return pcv.apply_mask(np_img, img_mask, 'white')


def roi_objects(img: Image.Image):
    np_img = np.asarray(img)
    img_mask = gray_scale(img)
    y_min, y_max, x_min, x_max = get_roi_rectangle(img_mask)

    new_img = np_img.copy()
    new_img[img_mask > 0] = GREEN
    thickness = 2
    new_img[y_min-thickness:y_min, x_min-thickness:x_max + thickness] = BLUE
    new_img[y_max:y_max+thickness,  x_min-thickness:x_max + thickness] = BLUE
    new_img[y_min:y_max, x_min-thickness:x_min] = BLUE
    new_img[y_min:y_max, x_max:x_max+thickness] = BLUE
    return new_img


def analyze_objects(img: Image.Image):
    pcv.params.line_thickness = 2
    np_img = np.asarray(img)
    img_mask = gray_scale(img)
    y_min, y_max, x_min, x_max = get_roi_rectangle(img_mask)
    roi = pcv.roi.rectangle(img, x_min, y_min, y_max - y_min, x_max - x_min)
    filtered_mask = pcv.roi.filter(mask=img_mask, roi=roi, roi_type="partial")
    return pcv.analyze.size(img=np_img, labeled_mask=filtered_mask, n_labels=1)


transforms = {
    "g_blur": ("Gaussian blur", gaussian_blur),
    "mask": ("Mask", mask),
    "roi": ("Roi objects", roi_objects),
    "analyze": ("Analyze objects", analyze_objects),
    "landmarks": ("Pseudolandmarks", pseudolandmarks),
}
