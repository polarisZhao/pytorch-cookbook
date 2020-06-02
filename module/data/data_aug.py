#!usr/bin/env python3
# -*- coding:utf-8 -*-

# data aug
# author: zhaozhichao
# reference:
# https://github.com/apache/incubator-mxnet/tree/6f9a67901362a794e3c022dd75daf8a516760fea/python/mxnet/image
# http://scipy-lectures.org/advanced/image_processing/
# https://blog.csdn.net/lwplwf/article/details/85776309
# https://blog.csdn.net/u011995719/article/details/85107009
# https://github.com/pytorch/vision/tree/master/torchvision/transforms
# https://github.com/mirzaevinom/data_science_bowl_2018/blob/master/codes/augment_preprocess.py
# https://github.com/jacobkie/2018DSB/blob/07df7d385f23a2272d8258351d680b037705ce3c/script_final/preprocess.py
# https://github.com/selimsef/dsb2018_topcoders/blob/master/albu/src/augmentations/functional.py

import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# FancyPCA
def FancyPCA(img):
    """
    """
    h, w, c = img.shape
    img = np.reshape(img, (h * w, c)).astype('float32')
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    img = (img - mean) / std

    cov = np.cov(img, rowvar=False)
    lambdas, p = np.linalg.eig(cov)
    alphas = np.random.normal(0, 0.1, c)
    pca_img = img + np.dot(p, alphas*lambdas)

    pca_color_img = pca_img * std + mean
    pca_color_img = np.maximum(np.minimum(pca_color_img, 255), 0)
    return pca_color_img.reshape(h, w, c).astype(np.uint8)

# Flip and Rotation
def random_horizontal_flip(img, p):
    """
        img : Image to be horizontal flipped
        p: probability that image should be horizontal flipped.
    """
    if np.random.random() < p:
        img = np.fliplr(img) 
    return img

def random_vertical_flip(img, p):
    """
        img : Image to be vertical flipped
        p: probability that image should be vertical flipped.
    """
    if np.random.random() < p:
        img = np.flipud(img)
    return img

def random_rotate90(img, p):
    """
        img : Image to be random rotated
        p: probability that image should be random rotated.
    """
    if np.random.random() < p:
        angle = np.random.randint(1, 3) * 90

        if angle == 90:
            img = img.transpose(1,0,2)
            img = np.fliplr(img)

        elif angle == 180:
        	img = np.rot90(img, 2)

        elif angle == 270:
            img = img.transpose(1,0,2)
            img = np.flipud(img)
    return img

def rotate(img, angle):
    """
        img : Image to be rotated
        angle(degree measure): angle to be rotated
    """
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img = cv2.warpAffine(img, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img

def shift_scale_rotate(img, angle, scale, dx, dy):
    """
    img : Image to be affine transformation
    angle(degree measure): angle to be rotated(15, 30)
    scale: sclae(1.1/1.2/1.3)
    dx, dy: offset, Compared to the original image

    """
    height, width = img.shape[:2]

    angle = np.random.uniform(-angle, angle)
    scale = np.random.uniform(1.0/scale, scale)

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + \
        np.array([width/2+dx*width, height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return img


# color: brightness, contrast, saturation : Done
def random_brightness(img, brightness):
    """
    brightness : float, The brightness jitter ratio range, [0, 1]
    """
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_contrast(img, contrast):
    """
        contrast : The contrast jitter ratio range, [0, 1]
    """
    coef = np.array([[[0.114, 0.587,  0.299]]])   # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha*img  + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_saturation(img, saturation):
    coef = np.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray  = img * coef
    gray  = np.sum(gray,axis=2, keepdims=True)
    img = alpha*img  + (1.0 - alpha)*gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_color(img, brightness, contrast, saturation):
    """
        brightness : The brightness jitter ratio range, [0, 1]
        contrast : The contrast jitter ratio range, [0, 1]
        saturation : The saturation jitter ratio range, [0, 1]
    """
    if brightness > 0:
        img = random_brightness(img, brightness)
    if contrast > 0:
        img = random_contrast(img, contrast)
    if saturation > 0:
        img = random_saturation(img, saturation)
    return img

def random_hue(image, hue):
    """
     The hue jitter ratio range, [0, 1]
    """
    h = int(np.random.uniform(-hue, hue)*180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


# add noise
def random_noise(img, limit=[0, 0.1], p=1):
    if np.random.random() < p:
        H,W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H,W))*255

        img = img + noise[:,:,np.newaxis]*np.array([1,1,1])
        img = np.clip(img, 0, 255).astype(np.uint8)
        
    return img


# crop and resize
def random_crop(img, size):
    """
    size: (tuple) (new_w, new_h)
          value:   0.9*W > new_w > 0.8*W
                   0.9*H > new_h > 0.8*H
    """
    H, W = img.shape[:2]
    new_w, new_h = size
    assert(H > new_h)
    assert(W > new_w)

    x0 = np.random.choice(W-new_w) if W!=new_w else 0
    y0 = np.random.choice(H-new_h) if H!=new_h else 0

    if (new_w, new_h) != (W, H):
        img = img[y0:y0+new_h, x0:x0+new_w, :] 

    return img

def center_crop(img, size):
    """
    size: (tuple) (new_w, new_h)
    """
    H, W = img.shape[:2]
    new_w, new_h = size

    x0 = (W - new_w) // 2
    y0 = (H - new_h) // 2

    if (new_w, new_h) != (W, H):
        img = img[y0:y0+new_h, x0:x0+new_w] 

    return img

def _get_interp_method(interp, sizes=()):
    """
        interpolation method for all resizing operations
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
    sizes : tuple of int
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp

def resize(img, size, interp=2):
    h, w = img.shape[:2]

    if h > w:
        new_h, new_w = size * h // w, size
    else:
        new_h, new_w = size, size * w // h
    return cv2.resize(img, (new_w, new_h), _get_interp_method(interp, (h, w, new_h, new_w)))

def elastic_transform_fast(img, alpha=2, sigma=100, alpha_affine=100, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    shape = img.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(
        img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = np.float32(gaussian_filter(
        (random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
    dy = np.float32(gaussian_filter(
        (random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


if __name__ == '__main__':
    img_path = '/home/zhaozhichao/Desktop/3-workspace/FACE_MODE_3DDFA/example/b24.jpg'
    img = cv2.imread(img_path)
    cv2.imshow("origin img", img)
    if len(img.shape) == 2:
        w, h = img.shape[:2]
        img = img.reshape(w, h, 1)
    im2 = FancyPCA(img)
    cv2.imshow("im2", im2)
    cv2.waitKey() 