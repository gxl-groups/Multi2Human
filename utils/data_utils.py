
import imageio
import yaml
import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor)
import os
import os.path
import random

import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)

    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)

    def __len__(self):
        return self.length


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_default_dataset_paths():
    with open("datasets.yml") as yaml_file:
        read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    paths = {}
    for i in range(len(read_data)):
        paths[read_data[i]["dataset"]] = read_data[i]["path"]

    return paths


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_datasets(
    dataset_name,
    img_size,
    get_val_dataset=False,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
):
    _image_fnames = []
    ann_dir='/home/user/wz/second_project/Multi2Human/deepfashion_data/texture_ann/train'
    assert os.path.exists(f'{ann_dir}/upper_fused.txt')
    for idx, row in enumerate(
            open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
        annotations = row.split()
        _image_fnames.append(annotations[0])
    transform = Compose([Resize(img_size), ToTensor()])
    transform_with_flip = Compose([Resize(img_size), RandomHorizontalFlip(p=1.0), ToTensor()])

    default_paths = get_default_dataset_paths()

    if dataset_name in default_paths:
        dataset_path = default_paths[dataset_name]
    elif dataset_name == "Deepfashion": #
        if custom_dataset_path:
            dataset_path = '/home/user/wz/second_project/Multi2Human/deepfashion_data/dataset'
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    elif dataset_name == "DeepfashionT": #
        if custom_dataset_path:
            dataset_path = '/home/user/wz/second_project/Multi2Human/deepfashion_data/dataset_test'
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    else:
        raise ValueError(f"Invalid dataset chosen: {dataset_name}. To use a custom dataset, set --dataset \
            flag to 'custom'.")


    if dataset_name ==  "Deepfashion"or "DeepfashionT":
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.ImageFolder(
                dataset_path,
                transform=transform_with_flip,
            )

        if get_val_dataset:
            train_dataset, val_dataset = train_val_split(train_dataset, train_val_split_ratio)
            if get_flipped:
                train_dataset_flip, _ = train_val_split(train_dataset_flip, train_val_split_ratio)

    if get_flipped:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flip])

    if not get_val_dataset:
        val_dataset = None

    return train_dataset,_image_fnames, val_dataset

def get_datasets1(
    dataset_name,
    img_size,
    get_val_dataset=False,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
):
    transform = Compose([Resize(img_size), ToTensor()])
    transform_with_flip = Compose([Resize(img_size), RandomHorizontalFlip(p=1.0), ToTensor()])

    default_paths = get_default_dataset_paths()

    if dataset_name in default_paths:
        dataset_path = default_paths[dataset_name]
    elif dataset_name == "Deepfashion": #
        if custom_dataset_path:
            dataset_path = '/home/user/wz/second_project/Multi2Human/deepfashion_data/dataset'
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    elif dataset_name == "DeepfashionT": #
        if custom_dataset_path:
            dataset_path = '/home/user/wz/second_project/Multi2Human/deepfashion_data/dataset_test'
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    else:
        raise ValueError(f"Invalid dataset chosen: {dataset_name}. To use a custom dataset, set --dataset \
            flag to 'custom'.")


    if dataset_name ==  "Deepfashion"or "DeepfashionT":
        # train_dataset = torchvision.datasets.ImageFolder(
        #     dataset_path,
        #     transform=transform,
        # )
        train_dataset = DeepFashionImageDataset(
            img_dir='/home/user/wz/second_project/Multi2Human/deepfashion_data/datasets/train_images',
            ann_dir='/home/user/wz/second_project/Multi2Human/deepfashion_data/texture_ann/train',
            xflip=True)
        return train_dataset




def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=16,
    drop_last=True,
    shuffle=True,
    get_val_dataloader=False,
):

    train_dataset,_image_fnames, val_dataset= get_datasets(
        dataset_name,
        img_size,
        get_flipped=get_flipped,
        get_val_dataset=get_val_dataloader,
        train_val_split_ratio=train_val_split_ratio,
        custom_dataset_path='/home/user/wz/second_project/Multi2Human/deepfashion_data/dataset',
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=None,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last
    )

    return train_loader,_image_fnames

class DeepFashionAttrSegmDataset(data.Dataset):

    def __init__(self,
                 img_dir,
                 segm_dir,
                 # pose_dir,
                 text_dir,
                 ann_dir,
                 downsample_factor=2,
                 xflip=False):
        self._img_path = img_dir
        # self._densepose_path = pose_dir
        self._segm_path = segm_dir
        self._image_fnames = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip



        # load attributes
        assert os.path.exists(f'{ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames.append(annotations[0])

        self.caption_dict = {}
        this_text_path = os.path.join(text_dir)
        with open(this_text_path, 'r') as f:
            caption_all = f.readlines()
            for c in caption_all:
                name =c[0:c.rfind(':')]
                # #取.之前
                # e=E[0:E.rfind('.')]  #print 'A2A935'
                # #取.之后
                text=c[c.rfind(':'):]
                text = text.replace(':','')
                # print('name',name)
                # print('text', text)
                self.caption_dict[name] = text
        #print(self.caption_dict[name])

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(self._img_path, fname) as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_image_token(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}'
        with self._open_file(self._img_path, fname) as f:
            image = torch.load(f)
        return image

    def _load_text(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        #fname = f'{fname[:-4]}'
        prompt = self.caption_dict[fname].replace('\n', '').lower()
        return prompt



    # def _load_densepose(self, raw_idx):
    #     fname = self._image_fnames[raw_idx]
    #     fname = f'{fname[:-4]}_densepose.png'
    #     with self._open_file(self._densepose_path, fname) as f:
    #         densepose = Image.open(f)
    #         if self.downsample_factor != 1:
    #             width, height = densepose.size
    #             width = width // self.downsample_factor
    #             height = height // self.downsample_factor
    #             densepose = densepose.resize(
    #                 size=(width, height), resample=Image.NEAREST)
    #         # channel-wise IUV order, [3, H, W]
    #         densepose = np.array(densepose)[:, :, 2:].transpose(2, 0, 1)
    #     return densepose.astype(np.float32)

    def _load_segm(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_segm.png'
        with self._open_file(self._segm_path, fname) as f:
            segm = Image.open(f)
            if self.downsample_factor != 1:
                width, height = segm.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                segm = segm.resize(
                    size=(width, height), resample=Image.NEAREST)
            segm = np.array(segm)
        segm = segm[:, :, np.newaxis].transpose(2, 0, 1)
        return segm.astype(np.float32)

    def __getitem__(self, index):
        image = self._load_image_token(index)

        #print('1',image)
        # pose = self._load_densepose(index)
        segm = self._load_segm(index)
        text = self._load_text(index)


        # flip
        # if self.xflip and random.random() > 0.5:
        #     # image = image[:, :, ::-1].copy()
        #     # pose = pose[:, :, ::-1].copy() # 1，512，256
        #     segm = segm[:, :, ::-1].copy()

        segm = torch.from_numpy(segm) # 1，512，256

        # pose = pose / 12. - 1
        #image = image / 127.5 - 1

        return_dict = {
            'image': image,
            # 'densepose': pose,
            'prompt':text,
            'segm': segm,
            'img_name': self._image_fnames[index]
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)


class DeepFashionAttrSegmDataset_test(data.Dataset):

    def __init__(self,
                 img_dir,
                 segm_dir,
                 # pose_dir,
                 text_dir,
                 ann_dir,
                 downsample_factor=2,
                 xflip=False):
        self._img_path = img_dir
        # self._densepose_path = pose_dir
        self._segm_path = segm_dir
        self._image_fnames = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip



        # load attributes
        assert os.path.exists(f'{ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames.append(annotations[0])

        self.caption_dict = {}
        this_text_path = os.path.join(text_dir)
        with open(this_text_path, 'r') as f:
            caption_all = f.readlines()
            for c in caption_all:
                name =c[0:c.rfind(':')]
                # #取.之前
                # e=E[0:E.rfind('.')]  #print 'A2A935'
                # #取.之后
                text=c[c.rfind(':'):]
                text = text.replace(':','')
                # print('name',name)
                # print('text', text)
                self.caption_dict[name] = text
        #print(self.caption_dict[name])

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(self._img_path, fname) as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_image_token(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}'
        with self._open_file(self._img_path, fname) as f:
            image = torch.load(f)
        return image

    def _load_text(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        #fname = f'{fname[:-4]}'
        prompt = self.caption_dict[fname].replace('\n', '').lower()
        return prompt



    # def _load_densepose(self, raw_idx):
    #     fname = self._image_fnames[raw_idx]
    #     fname = f'{fname[:-4]}_densepose.png'
    #     with self._open_file(self._densepose_path, fname) as f:
    #         densepose = Image.open(f)
    #         if self.downsample_factor != 1:
    #             width, height = densepose.size
    #             width = width // self.downsample_factor
    #             height = height // self.downsample_factor
    #             densepose = densepose.resize(
    #                 size=(width, height), resample=Image.NEAREST)
    #         # channel-wise IUV order, [3, H, W]
    #         densepose = np.array(densepose)[:, :, 2:].transpose(2, 0, 1)
    #     return densepose.astype(np.float32)

    def _load_segm(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_segm.png'
        with self._open_file(self._segm_path, fname) as f:
            segm = Image.open(f)
            if self.downsample_factor != 1:
                width, height = segm.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                segm = segm.resize(
                    size=(width, height), resample=Image.NEAREST)
            segm = np.array(segm)
        segm = segm[:, :, np.newaxis].transpose(2, 0, 1)
        return segm.astype(np.float32)

    def __getitem__(self, index):
#        image = self._load_image_token(index)

        #print('1',image)
        # pose = self._load_densepose(index)
        segm = self._load_segm(index)
        text = self._load_text(index)


        # flip
        # if self.xflip and random.random() > 0.5:
        #     # image = image[:, :, ::-1].copy()
        #     # pose = pose[:, :, ::-1].copy() # 1，512，256
        #     segm = segm[:, :, ::-1].copy()

        segm = torch.from_numpy(segm) # 1，512，256

        # pose = pose / 12. - 1
        #image = image / 127.5 - 1

        return_dict = {
#            'image': image,
            # 'densepose': pose,
            'prompt':text,
            'segm': segm,
            'img_name': self._image_fnames[index]
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)


class DeepFashionImageDataset(data.Dataset):

    def __init__(self,
                 img_dir,
                 ann_dir,
                 downsample_factor=2,
                 xflip=False):
        self._img_path = img_dir
        self._image_fnames = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip

        # load attributes
        assert os.path.exists(f'{ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames.append(annotations[0])

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(self._img_path, fname) as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image


    def __getitem__(self, index):
        image = self._load_raw_image(index)

        # flip
        # if self.xflip and random.random() > 0.5:
        #     assert image.ndim == 3  # CHW
        #     image = image[:, :, ::-1].copy()

        image = torch.from_numpy(image) # 3，512，256
        # pose = pose / 12. - 1
        image = image / 127.5 - 1

        return_dict = {
            'image': image,
            'img_name': self._image_fnames[index]
        }
        # print(self._image_fnames[index])
        # print('23456',self._image_fnames[index])
        return return_dict

    def __len__(self):
        return len(self._image_fnames)



class DeepFashionAttrSegmDataset1(data.Dataset):

    def __init__(self,
                 img_dir,
                 segm_dir,
                 # pose_dir,
                 text_dir,
                 ann_dir,
                 downsample_factor=2,
                 xflip=False):
        self._img_path = img_dir
        # self._densepose_path = pose_dir
        self._segm_path = segm_dir
        self._image_fnames = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip



        # load attributes
        assert os.path.exists(f'{ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames.append(annotations[0])

        self.caption_dict = {}
        this_text_path = os.path.join(text_dir)
        with open(this_text_path, 'r') as f:
            caption_all = f.readlines()
            for c in caption_all:
                name =c[0:c.rfind(':')]
                # #取.之前
                # e=E[0:E.rfind('.')]  #print 'A2A935'
                # #取.之后
                text=c[c.rfind(':'):]
                text = text.replace(':','')
                # print('name',name)
                # print('text', text)
                self.caption_dict[name] = text
        #print(self.caption_dict[name])

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(self._img_path, fname) as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_image_token(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}'
        with self._open_file(self._img_path, fname) as f:
            image = torch.load(f)
        return image


    def __getitem__(self, index):
        image = self._load_image_token(index)

        image = image / 127.5 - 1

        return_dict = {
            'image': image,
            'img_name': self._image_fnames[index]
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)