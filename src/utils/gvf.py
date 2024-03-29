#!env python
# -*- coding: utf-8 -*-
# GVF(Gradient Vector Flow) of 2D image.
# implemented by t-suzuki.
# License: Public Domain
#
# reference:
# Chenyang Xu, et al. "Snakes, Shpes, and Gradient Vector Flow", IEEE TRANACTIONS ON IMAGE PROCESSING, VOL. 7, NO. 3, MARCH 1998

import sys
import skimage.color
import skimage.data
import skimage.transform
import numpy as np

def gradient_vector_flow(img, fx, fy, mu, dx=1.0, dy=1.0, verbose=True):
    '''calc gradient vector flow of input gradient field fx, fy'''
    # calc some coefficients.
    b = fx**2.0 + fy**2.0
    c1, c2 = b*fx, b*fy
    # calc dt from scaling parameter r.
    r = 0.25 # (17) r < 1/4 required for convergence.
    dt = dx*dy/(r*mu)
    # max iteration
    N = int(max(1, np.sqrt(img.shape[0]*img.shape[1])))
    # initialize u(x, y), v(x, y) by the input.
    curr_u = fx
    curr_v = fy
    def laplacian(m):
        return np.hstack([m[:, 0:1], m[:, :-1]]) + np.hstack([m[:, 1:], m[:, -2:-1]]) \
                + np.vstack([m[0:1, :], m[:-1, :]]) + np.vstack([m[1:, :], m[-2:-1, :]]) \
                - 4*m
    for i in range(N):
        next_u = (1.0 - b*dt)*curr_u + r*laplacian(curr_u) + c1*dt
        next_v = (1.0 - b*dt)*curr_v + r*laplacian(curr_v) + c2*dt
        curr_u, curr_v = next_u, next_v
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
    if verbose:
        sys.stdout.write('\n')
    return curr_u, curr_v

def prepare_img_for_gvf(img, sigma):
    def prepare_img(img):
        def add_border(img, width):
            h, w = img.shape
            val = img[:, 0].mean() + img[:, -1].mean() + img[0, :].mean() + img[-1, :].mean()
            res = np.zeros((h + width*2, w + width*2), dtype=img.dtype) + val
            res[width:h+width, width:w+width] = img
            res[:width, :] = res[width, :][np.newaxis, :]
            res[:, :width] = res[:, width][:, np.newaxis]
            res[h+width:, :] = res[h+width-1, :][np.newaxis, :]
            res[:, w+width:] = res[:, w+width-1][:, np.newaxis]
            return res  

        img = skimage.color.rgb2gray(img)
        img = img.astype(np.float32) / 255.0
        img = add_border(img, 32)
        return img

    def edge_map(img, sigma):
        blur = skimage.filters.gaussian(img, sigma)
        return skimage.filters.sobel(blur)


    img = prepare_img(img)
    edge = edge_map(img, sigma)

    return img, edge

def gradient_field(im):
    im = skimage.filters.gaussian(im, 1.0)
    gradx = np.hstack([im[:, 1:], im[:, -2:-1]]) - np.hstack([im[:, 0:1], im[:, :-1]]) 
    grady = np.vstack([im[1:, :], im[-2:-1, :]]) - np.vstack([im[0:1, :], im[:-1, :]]) 
    return gradx, grady


