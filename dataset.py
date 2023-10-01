from torch.utils.data import Dataset
from tensorflow.keras.preprocessing import image
#from tensorflow import image 
import numpy as np
from utils import np2tensor
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch, pdb
import tensorflow as tf


class FIW(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None):

        self.sample_path=sample_path
        self.transform=transform
        self.sample_list=self.load_sample()
        self.bias=0

    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                # pdb.set_trace()
                tmp = line.split(' ')
                sample_list.append([tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]])
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        #img = tf.image.resize(path, [112, 112])
        return img

    def set_bias(self,bias):
        self.bias=bias

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample = self.sample_list[item+self.bias]
        img1,img2=self.read_image(sample[1]),self.read_image(sample[2])        
        if self.transform is not None:
            img1,img2 = self.transform(img1),self.transform(img2)
        img1, img2 = np2tensor(self.preprocess(np.array(img1, dtype=float))), \
                     np2tensor(self.preprocess(np.array(img2, dtype=float)))

        # pdb.set_trace()
        label = np2tensor(np.array(sample[4], dtype=float))
        # pdb.set_trace()
        # kin_label = np2tensor(np.array(sample[3], dtype=float))
        kin_label = sample[3]
        # quality = sample[5]
        return img1, img2, label, kin_label


class FIW2(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None):

        self.sample_path=sample_path
        self.transform=transform
        self.sample_list=self.load_sample()
        self.bias=0

    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                # pdb.set_trace()
                tmp = line.split(' ')
                sample_list.append([tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]])
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        #img = tf.image.resize(path, [112, 112])
        return img

    def set_bias(self,bias):
        self.bias=bias

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample = self.sample_list[item+self.bias]
        img1,img2=self.read_image(sample[1]),self.read_image(sample[2])        
        if self.transform is not None:
            img1,img2 = self.transform(img1),self.transform(img2)
        img1, img2 = np2tensor(self.preprocess(np.array(img1, dtype=float))), \
                     np2tensor(self.preprocess(np.array(img2, dtype=float)))

        # pdb.set_trace()
        label = np2tensor(np.array(sample[4], dtype=float))
        # pdb.set_trace()
        # kin_label = np2tensor(np.array(sample[3], dtype=float))
        kin_label = sample[3]
        quality = np2tensor(np.array(sample[5], dtype=float))
        return img1, img2, label, kin_label, quality


class FIW_R(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None):

        self.sample_path=sample_path
        self.transform=transform
        self.sample_list=self.load_sample()
        self.bias=0

    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                # pdb.set_trace()
                tmp = line.split(' ')
                sample_list.append([tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]])
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        #img = tf.image.resize(path, [112, 112])
        return img

    def set_bias(self,bias):
        self.bias=bias

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample = self.sample_list[item+self.bias]
        img1,img2=self.read_image(sample[1]),self.read_image(sample[2])
        # pdb.set_trace()
        img1_1, img2_1 = align.get_aligned_face(sample[1]), align.get_aligned_face(sample[2])
        
        if self.transform is not None:
            img1,img2 = self.transform(img1),self.transform(img2)
        img1, img2 = np2tensor(self.preprocess(np.array(img1, dtype=float))), \
                     np2tensor(self.preprocess(np.array(img2, dtype=float)))
        # pdb.set_trace()
        try:
            img1_1 = np2tensor(self.preprocess(np.array(img1_1, dtype=float)))
        except:
            img1_1 = img1
        try:
            img2_1 = np2tensor(self.preprocess(np.array(img2_1, dtype=float)))
        except:
            img2_1 = img2
        # pdb.set_trace()
        label = np2tensor(np.array(sample[4], dtype=float))
        # pdb.set_trace()
        # kin_label = np2tensor(np.array(sample[3], dtype=float))
        kin_label = sample[3]
        return img1_1, img2_1, label, kin_label
