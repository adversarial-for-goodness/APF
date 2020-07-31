import numpy as np
import os
import pickle
from scipy import misc
import io
import imageio
from PIL import Image

def create_lfw_npy(path='/data/jiaming/datasets/faces/faces_emore/lfw.bin',image_size=112):
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    for bin in bins:
        img = imageio.imread(io.BytesIO(bin))
        img = np.array(Image.fromarray(img).resize([image_size, image_size]))

        # img = img[s:s+image_size, s:s+image_size, :]
        img = img / 127.5 - 1.0
        images[cnt] = img
        cnt += 1
    print('done!')
    print('cnt number is ' + str(cnt))
    images = images
    images_copy = np.zeros((6000,image_size,image_size,3))
    for i in range(10):
        images_copy[600*i:600*(i+1)] = images[600*i*2:600*(i*2+1)]
    return images_copy

def create_cfp_npy(path='/data/jiaming/datasets/faces/faces_emore/cfp_fp.bin',image_size=112):
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    imgs = np.zeros((7000,image_size,image_size,3))
    for i,issame in enumerate(issame_list):
        img0 = misc.imread(io.BytesIO(bins[2*i]))
        img1 = misc.imread(io.BytesIO(bins[2*i+1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            imgs[cnt:cnt+1] = img0
            imgs[cnt+1:cnt+2] = img1
            cnt += 2
    print('done!')
    print(cnt)
    return imgs

def create_agedb_npy(path='/data/jiaming/datasets/faces/faces_emore/agedb_30.bin',image_size=112):
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    imgs = np.zeros((6000,image_size,image_size,3))
    for i,issame in enumerate(issame_list):
        img0 = misc.imread(io.BytesIO(bins[2*i]))
        img1 = misc.imread(io.BytesIO(bins[2*i+1]))
        img0 = misc.imresize(img0, [image_size, image_size])
        img1 = misc.imresize(img1, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]

        img0 = img0 / 127.5 - 1.0
        img1 = img1 / 127.5 - 1.0
        if (issame):
            # print(issame)
            imgs[cnt:cnt+1] = img0
            imgs[cnt+1:cnt+2] = img1
            cnt += 2
    print('done!')
    print(cnt)
    return imgs
