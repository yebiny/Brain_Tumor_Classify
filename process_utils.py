import pydicom as dcm
import numpy as np
import cv2
import os, glob
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt

def get_dcm_files(TRAIN_DIR, p_id, img_type):
    img_files = glob.glob("%s/%s/%s/*dcm"%(TRAIN_DIR, p_id, img_type))
    img_files.sort(key=lambda x: int(x.split('Image-')[1].split('.')[0]))
    return img_files

def load_dcm_imgs(path, img_size=256, voi_lut=True, rotate=None, flip=None):
    dicom = dcm.read_file(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    if rotate!=None:
        rot_choices = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        data = cv2.rotate(data, rot_choices[rotate])
    if flip!=None:
        data = cv2.flip(data, flip)
        
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dcm_imgs_3d(dcm_files, num_imgs=128, img_size=256, rotate=None, flip=None):

    middle = len(dcm_files)//2
    num_imgs2 = num_imgs//2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(dcm_files), middle + num_imgs2)
    img3d = np.stack([load_dcm_imgs(f, img_size=img_size, rotate=rotate, flip=flip) for f in dcm_files[p1:p2]]).T 

    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d,  n_zero), axis = -1)

    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)

    return img3d

def img_2_patches(img, img_size=256, crop_size=64):
    n_patches = int(img_size/crop_size)
    patches=[]
    for n1 in range(n_patches):
        start1 = n1*crop_size
        end1 = start1+crop_size

        for n2 in range(n_patches):
            start2 = n2*crop_size
            end2 = start2+crop_size

            patch = img[start1:end1, start2:end2]
            patches.append(patch)
    return np.array(patches)

def plot_patches(patches, ch=30):
    n = int(np.sqrt(patches.shape[0]))
    plt.figure(figsize=(4,4))
    for i, p in enumerate(patches):
        plt.subplot(n,n,i+1)
        plt.imshow(p[:,:,ch])
        plt.xticks([])
        plt.yticks([])
    plt.show()
