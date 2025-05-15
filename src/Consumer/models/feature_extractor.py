import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
import cv2

class HOGFeatureExtractor:
    def __init__(self):
        self.params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }

    def extract(self, img):
        if len(img.shape) == 3:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return hog(img, **self.params)


class LBPFeatureExtractor:
    def __init__(self):
        self.params = {
            'P': 8,
            'R': 1,
            'method': 'uniform'
        }

    def extract(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(img, **self.params)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9), density=True)
        return hist


class GLCMFeatureExtractor:
    def __init__(self):
        self.params = {
            'distances': [1],
            'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
            'levels': 256,
            'symmetric': True,
            'normed': True
        }
        self.props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']

    def extract(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(img, **self.params)
        return np.concatenate([graycoprops(glcm, p).flatten() for p in self.props])


class GaborFeatureExtractor:
    def __init__(self):
        self.thetas = np.linspace(0, np.pi, 4, endpoint=False)
        self.sigmas = [1, 3]
        self.lambdas = [np.pi/4, np.pi/2]

    def extract(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features = []
        for theta in self.thetas:
            for sigma in self.sigmas:
                for lambd in self.lambdas:
                    kernel = np.real(gabor_kernel(frequency=1.0/lambd, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    filtered = convolve(img.astype(np.float32), kernel, mode='reflect')
                    features.append(filtered.mean())
                    features.append(filtered.var())
        return np.array(features)


class SIFTFeatureExtractor:
    def __init__(self, max_features=128):
        self.sift = cv2.SIFT_create()
        self.max_features = max_features

    def extract(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        if descriptors.shape[0] < self.max_features:
            pad = np.zeros((self.max_features - descriptors.shape[0], 128), dtype=np.float32)
            descriptors = np.vstack((descriptors, pad))
        else:
            descriptors = descriptors[:self.max_features]
        return descriptors.flatten()