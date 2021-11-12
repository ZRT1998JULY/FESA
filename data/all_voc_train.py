from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import torch
import PIL.Image as Image
import numpy as np
from torchvision.utils import save_image
from config import settings

#random.seed(1385)

class all_voc_train():

    """Face Landmarks dataset."""

    def __init__(self, args, size, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = 20
        self.group = args.group
        self.num_folds = args.num_folds
        self.mode = mode
        #self.binary_map_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train') #val
        if mode == 'train':
            self.data_list_file = '/media/meng2/disk2/ZRT/dataset/VOC2010/train_for_component_aug.txt'
            self.img_dir = '/media/meng2/disk2/ZRT/dataset/VOC2010/JPEGImages'
            self.mask_dir = '/media/meng2/disk2/ZRT/dataset/VOC2010/color_segmentation_14'
            self.part_mask_dir = '/media/meng2/disk2/ZRT/dataset/VOC2010/color_part_segmentation_28'
        elif mode == 'test_voc':
            self.data_list_file = '/media/meng2/disk2/ZRT/dataset/VOC2010/val_for_component_aug.txt'
            #self.data_list_file = '/media/meng2/disk2/ZRT/dataset/VOC2010/train_for_component_aug.txt'
            #self.data_list_file = '/media/meng2/disk2/ZRT/dataset/VOC2010/voc_cat/voc_dog.txt' # 4:0.72 +  5:0.80 -- m : 0.68
            self.img_dir = '/media/meng2/disk2/ZRT/dataset/VOC2010/JPEGImages'
            self.mask_dir = '/media/meng2/disk2/ZRT/dataset/VOC2010/color_segmentation_14'
            self.part_mask_dir = '/media/meng2/disk2/ZRT/dataset/VOC2010/color_part_segmentation_28'
        elif mode == 'test_coco':
            self.data_list_file = '/media/meng2/disk2/ZRT/dataset/COCO2017/coco_cat/coco_giraffe.txt' # 2:0.74  ---  0.76
            #self.data_list_file = '/media/meng2/disk2/ZRT/dataset/COCO2017/coco_classification_train_sort.txt'
            #self.data_list_file = '/media/meng2/disk2/ZRT/dataset/COCO2017/coco_classification_val_sort.txt'
            self.img_dir = '/media/meng2/disk2/ZRT/dataset/COCO2017/trainval2017'
            self.mask_dir = '/media/meng2/disk2/ZRT/dataset/COCO2017/coco_segmentation_20_new'
        else:
            print('running mode error!')


        #self.binary_mask_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train/')

        self.train_list = self.get_train_list()

        self.transform = transform
        self.count = 0
        self.random_generator = random.Random()
        self.size = size
        #self.random_generator.shuffle(self.list_splite)
        #self.random_generator.seed(1385)


    def get_train_list(self):
        f = open(self.data_list_file)
        data_list = []
        while True:
            img_name = f.readline()[:-1]
            if img_name == '':
                break
            data_list.append(img_name)
        return data_list

    def read_img(self, name):
        path = os.path.join(self.img_dir, name + '.jpg')
        img = Image.open(path)
        return img

    def read_mask(self, name):
        path = os.path.join(self.mask_dir, name + '.png')
        mask = Image.open(path)
        return mask

    def read_part_mask(self, name):
        path = os.path.join(self.part_mask_dir, name + '.png')
        mask = Image.open(path)
        return mask


    def load_frame(self, img_name):
        img = self.read_img(img_name)
        mask = self.read_mask(img_name)
        if self.mode == 'train' or self.mode =='test_voc':
            part_mask = self.read_part_mask(img_name)
            return img, mask, part_mask
        elif self.mode =='test_coco':
            return img, mask

    def get_class_idx(self, mask):
        class_idx_list = np.sort(np.unique(mask))
        zero_index = np.argwhere(class_idx_list == 0)
        class_idx_list = np.delete(class_idx_list, zero_index)
        return class_idx_list

    def mask_convert(self, mask):
        mask = np.asarray(mask).astype(np.float32)
        idx_list = np.unique(mask).astype(np.uint8).tolist()
        cat_idx = np.zeros((21))
        cat_idx[idx_list] = 1.0
        return mask, cat_idx

    def part_mask_convert(self, part_mask):
        mask_return = np.asarray(part_mask).astype(np.float32)
        return mask_return

    def __len__(self):
        # return len(self.image_list)
        return  len(self.train_list)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode =='test_voc':
            img, mask, part_mask = self.load_frame(self.train_list[idx])
            mask, cat_idx = self.mask_convert(mask)
            part_mask = self.part_mask_convert(part_mask)
            img, mask, part_mask = self.transform(img, mask, part_mask)
            return img, mask, part_mask, cat_idx

        elif self.mode == 'test_coco':
            name = self.train_list[idx].split(" ")[0]
            img, mask = self.load_frame(name)
            mask, cat_idx = self.mask_convert(mask)
            img, mask = self.transform(img, mask, None)
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            return img, mask, torch.tensor(0), cat_idx