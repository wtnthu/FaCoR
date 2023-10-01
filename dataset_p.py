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
                #pdb.set_trace()
                tmp = line.split('\t')
                # pdb.set_trace()
                sample_list.append([tmp[0], tmp[1],  tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]])
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
        # pdb.set_trace()
        sample = self.sample_list[item+self.bias]
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        if self.transform is not None:
            img1,img2 = self.transform(img1),self.transform(img2)
        img1, img2 = np2tensor(self.preprocess(np.array(img1, dtype=float))), \
                     np2tensor(self.preprocess(np.array(img2, dtype=float)))
        label = np2tensor(np.array(sample[2], dtype=float))
        kin_label = np2tensor(np.array(sample[4], dtype=float))
        id1 = np2tensor(np.array(sample[5], dtype=float))
        id2 = np2tensor(np.array(sample[6], dtype=float))
        return img1, img2, label, kin_label, id1, id2
