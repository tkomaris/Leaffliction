from PIL import Image, ImageOps
import math

fillcolor = "#fff"
scalingFactor = 0.15


def transform(x, y):
    y = y + 30 * math.sin(x / 80)
    x = x + 30 * math.sin(y / 80)
    return x, y


def transform_rectangle(x0, y0, x1, y1):
    return (*transform(x0, y0),
            *transform(x0, y1),
            *transform(x1, y1),
            *transform(x1, y0),
            )


def getWaveMesh(img):
    gridspace = 20
    target_grid = []
    for x in range(0, img.size[0], gridspace):
        for y in range(0, img.size[1], gridspace):
            target_grid.append((x, y, x + gridspace, y + gridspace))

    source_grid = [transform_rectangle(*rect) for rect in target_grid]
    return [t for t in zip(target_grid, source_grid)]


def crop(img: Image.Image):
    return ImageOps.crop(img, scalingFactor * img.size[0]).resize(img.size)


def flip(img: Image.Image):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def rotate(img: Image.Image):
    return img.rotate(-20, expand=True, fillcolor=fillcolor)


def blur(img: Image.Image):
    return ImageOps.scale(img, 0.3).resize(img.size)


def contrast(img: Image.Image):
    return ImageOps.autocontrast(img.convert("RGB"), cutoff=5)


def deform(img: Image.Image):
    return img.transform(
        img.size, Image.Transform.MESH,
        [(
            (0, 0, img.size[0], img.size[1]),
            (0, -50,
             -50, img.size[1],
             img.size[0] + 20, img.size[1] + 90,
             img.size[0], 0)
        )],
        fillcolor=fillcolor
    )


def wave(img: Image.Image):
    return img.transform(
        img.size, Image.Transform.MESH, getWaveMesh(img), fillcolor=fillcolor
    )


auguments = {
    "Flip": flip,
    "Rotate": rotate,
    "Blur": blur,
    "Contrast": contrast,
    "Crop": crop,
    "Deform": deform,
    "Wave": wave
    }
