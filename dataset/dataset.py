from io import FileIO
import sys, os, glob
from copy import deepcopy
from PIL import Image
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import pickle
import gzip
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from copy import copy
from abc import ABCMeta, abstractclassmethod
from dataset_wrapper import LabelsQueriable

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, labels_set, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in labels_set:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, file_names in sorted(os.walk(d)):
            for file_name in sorted(file_names):
                if has_file_allowed_extension(file_name, extensions):
                    path = os.path.join(root, file_name)
                    item = (path, target)
                    images.append(item)

    return images
    
def open_split_train_validate(dataset, val_percent=0.1, class_by='cultivar', seed=None , val_labels=None):
    np.random.seed(seed)
    train_ds = deepcopy(dataset)
    val_ds = deepcopy(train_ds)
    labels_set = set(train_ds.labels[class_by])
    num_class_val = round(len(labels_set) * val_percent)
    if num_class_val == 0:
        val_ds.image_paths = []
        val_ds.labels = {}
        return train_ds, val_ds
    if val_labels is None:
        rs = np.random.RandomState(seed)
        val_labels = set(rs.choice(list(labels_set), num_class_val, replace=False))
    else:
        val_labels = set(val_labels)
    train_labels = labels_set - val_labels
    train_filtered_idx = [i for i, c in enumerate(train_ds.labels[class_by]) if c in train_labels]
    for key in train_ds.labels.keys():
        train_ds.labels[key] = [train_ds.labels[key][i] for i in train_filtered_idx]
    train_ds.image_paths = [train_ds.image_paths[i] for i in train_filtered_idx]
    val_filtered_idx = [i for i, c in enumerate(val_ds.labels[class_by]) if c in val_labels]
    for key in val_ds.labels.keys():
        val_ds.labels[key] = [val_ds.labels[key][i] for i in val_filtered_idx]
    val_ds.image_paths = [val_ds.image_paths[i] for i in val_filtered_idx]
    return train_ds, val_ds

class OPENScanner3dDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, file_type='depth', file_list=None,
                 start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, 
                 transform=None):
        if file_type == 'depth':
            self.image_file_suffix = 'p.png'
        elif file_type == 'reflectance':
            self.image_file_suffix = 'g.png'
        elif file_type == 'xyz':
            self.image_file_suffix = 'xyz.npy.gz'
        else:
            raise ValueError('{} file type not exist.'.format(file_type))
        self.open_dataset_root = open_dataset_root
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.file_list = file_list
        self._parse_file_list()
        self.image_paths, self.labels = self._make_dataset() 

    def _parse_file_list(self):
        if self.file_list is not None:
            if type(self.file_list) is str:
                if os.path.splitext(self.file_list)[1] == ".pkl":
                    with open(self.file_list, 'rb') as f:
                        self.file_list = pickle.load(f)
                else:
                    raise FileIO("Unknown file type")
            elif type(self.file_list) is list:
                pass
            else:
                raise TypeError(f"Unknown type of file_list, {type(self.file_list)}")        

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        if self.file_list is not None:
            file_path_list = [os.path.join(self.open_dataset_root, os.path.normpath(i)) for i in self.file_list]
        else:
            file_path_list = glob.glob(os.path.join(self.season_root, '**', '**', 'scanner3DTop', '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'cultivar': [], 'plot': [], 'scan_date': [], 'scanner': []}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            cultivar = path_split[-4]
            plot = path_split[-3]
            scan_date = datetime.strptime(path_split[-1][:20], '%Y-%m-%d__%H-%M-%S').date().toordinal()
            if 'east' in path_split[-1]:
                scanner = 'east'
            elif 'west' in path_split[-1]:
                scanner = 'west'
            else:
                continue
            paths.append(file_path)
            labels['cultivar'].append(cultivar)
            labels['plot'].append(plot)
            labels['scan_date'].append(scan_date)
            labels['scanner'].append(scanner)
        le = LabelEncoder()
        cultivar_int = le.fit_transform(labels['cultivar'])
        labels['cultivar_int'] = cultivar_int
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]

        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class OPENScanner3dSurfNormDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, file_list=None,
                 start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, transform=None):
        self.image_file_suffix = '.png'
        self.open_dataset_root = open_dataset_root
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.file_list = file_list
        self._parse_file_list()
        self.image_paths, self.labels = self._make_dataset() 

    def _parse_file_list(self):
        if self.file_list is not None:
            if type(self.file_list) is str:
                if os.path.splitext(self.file_list)[1] == ".pkl":
                    with open(self.file_list, 'rb') as f:
                        self.file_list = pickle.load(f)
                else:
                    raise FileIO("Unknown file type")
            elif type(self.file_list) is list:
                pass
            else:
                raise TypeError(f"Unknown type of file_list, {type(self.file_list)}") 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        if self.file_list is None:
            file_path_list = glob.glob(os.path.join(self.season_root, '**', '**', 'scanner3DTop_preprocessed', '*{}'.format(self.image_file_suffix)))
        else:
            file_path_list = [os.path.join(self.open_dataset_root, os.path.normpath(i)) for i in self.file_list]
        paths = []
        labels = {'cultivar': [], 'plot': [], 'scan_date': [], 'scanner': []}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            cultivar = path_split[-4]
            row, col, plot = path_split[-3].split('_')
            plot = int(plot)
            scan_date = datetime.strptime(path_split[-1][:20], '%Y-%m-%d__%H-%M-%S').date().toordinal()
            if 'east' in path_split[-1]:
                scanner = 'east'
            elif 'west' in path_split[-1]:
                scanner = 'west'
            else:
                continue
            paths.append(file_path)
            labels['cultivar'].append(cultivar)
            labels['plot'].append(plot)
            labels['scan_date'].append(scan_date)
            labels['scanner'].append(scanner)
        le = LabelEncoder()
        cultivar_int = le.fit_transform(labels['cultivar'])
        labels['cultivar_int'] = cultivar_int
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]

        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class OPENStereoDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, start_date=None, end_date=None, plots=None, exclude_date=None,
                 exclude_cultivar=None, image_file_suffix='.png', transform=None):
        self.open_dataset_root = open_dataset_root
        self.image_file_suffix = image_file_suffix
        self.season = season
        self.plots = plots
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.image_paths, self.labels = self._make_dataset() 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        sensor_name = 'stereoTop'
        if self.image_file_suffix == '.jpg':
            sensor_name += '_jpg'
        file_path_list = glob.glob(os.path.join(self.season_root, '**', '**', sensor_name, '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'cultivar': [], 'plot': [], 'scan_date': [], 'plot_row':[], 'plot_column':[]}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            cultivar = path_split[-4]
            plot_row = int(path_split[-3].split('_')[0])
            plot_column = int(path_split[-3].split('_')[1])
            plot = int(path_split[-3].split('_')[2])
            scan_date = datetime.strptime(path_split[-1][:20], '%Y-%m-%d__%H-%M-%S').date().toordinal()
            paths.append(file_path)
            labels['cultivar'].append(cultivar)
            labels['plot'].append(plot)
            labels['plot_row'].append(plot_row)
            labels['plot_column'].append(plot_column)
            labels['scan_date'].append(scan_date)
        # filter dataset by conditions
        # select plots
        if self.plots is not None:
            filtered_idx = [i for i, t in enumerate(labels['plot']) if t in self.plots]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<=t<=end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        le = LabelEncoder()
        cultivar_int = le.fit_transform(labels['cultivar'])
        labels['cultivar_int'] = cultivar_int
        plot_le = LabelEncoder()
        plot_cls_int = plot_le.fit_transform(labels['plot'])
        labels['plot_cls'] = plot_cls_int
        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class WheatTripletDataset(Dataset):
    def __init__(self, df_path=None, key_file_path = None, img_dir=None, transform = None):
        self.img_dir = img_dir
        self.data_df = pd.read_csv(df_path)
        self.key_file = pd.read_csv(key_file_path)
        self.transform = transform
        self.label_to_indices = {label: np.where(np.array(self.data_df.plot_id) == label)[0] for label in self.key_file['plot_id']}
        self.label_to_indices = {k: v for k, v in self.label_to_indices.items() if len(v)!=0}
        
    def __len__(self):
        return len(self.data_df.index)
    
    def __getitem__(self, index):
        # Anchor Image Data
        anchor_img_folder = self.data_df.folder[index]
        anchor_img_name = self.data_df.img_name[index]
        anchor_label = self.data_df.plot_id[index]
        anchor_img = Image.open(os.path.join(self.img_dir, anchor_img_folder, anchor_img_name))
        
        #  Find Negative Images
        negative_label = np.random.choice(list(set(list(self.label_to_indices.keys())) - set(list([anchor_label]))))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative_img_folder = self.data_df.folder[negative_idx]
        negative_img_name = self.data_df.img_name[negative_idx]
        negative_img = Image.open(os.path.join(self.img_dir, negative_img_folder, negative_img_name))
        
        #  Find Positive Image
        positive_idx = index
        while positive_idx == index:
            positive_idx = np.random.choice(self.label_to_indices[anchor_label])
        positive_img_folder = self.data_df.folder[positive_idx]
        positive_img_name = self.data_df.img_name[positive_idx]
        positive_img = Image.open(os.path.join(self.img_dir, positive_img_folder, positive_img_name))
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return (anchor_img, positive_img, negative_img), (anchor_label, anchor_label, negative_label)
    
class WheatSingleDataset(WheatTripletDataset):
    def __getitem__(self, index):
        img_folder = self.data_df.folder[index]
        img_name = self.data_df.img_name[index]
        label = self.data_df.plot_id[index]
        img = Image.open(os.path.join(self.img_dir, img_folder, img_name))
        if self.transform:
            img = self.transform(img)
        return img, label
    
def random_split_and_augment(my_dataset):
    torch.manual_seed(0)
    train_size = int(0.8 * len(my_dataset))
    trainset, testset = torch.utils.data.random_split(my_dataset, [train_size, len(my_dataset)-train_size])
    trainset.dataset = copy(my_dataset)
    
    trainset.dataset.transform = transforms.Compose(
        [transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(30),
         transforms.ToTensor()])
    
    testset.dataset.transform = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor()])
    return trainset, testset
    
class TripletWheatWithTimeDataset(Dataset):
    def __init__(self, look_up, cult_indx_dict, transform = None):
        self.data_df = look_up
        self.cul_indx_dict = cult_indx_dict
        self.transform = transforms.Compose(
        [transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15),
         transforms.ToTensor()])
    
    
    def __len__(self):
        return len(self.data_df.index)
    
    def __getitem__(self, index):
        # Anchor Image Data
        anchor_loc = self.data_df.location[index]
        anchor_img = Image.open(anchor_loc)
        anchor_cul = self.data_df.cultivar[index]
        anchor_dat = self.data_df.date[index]
        
        #  Find Negative Images
        negative_cul = np.random.choice(list(set(self.cult_list) - set([anchor_cul])))
        negative_idx = np.random.choice(self.cul_indx_dict[negative_cul])
        negative_loc = self.data_df.location[negative_idx]
        negative_img = Image.open(negative_loc)
        negative_dat = self.data_df.date[negative_idx]
        
        #  Find Positive Image
        positive_cul = anchor_cul
        positive_idx = index
        while positive_idx == index:
            positive_idx = np.random.choice(self.cul_indx_dict[anchor_cul])
        
        positive_loc = self.data_df.location[positive_idx]
        positive_img = Image.open(positive_loc)
        positive_dat = self.data_df.date[positive_idx]
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return (anchor_img, positive_img, negative_img), (anchor_dat, positive_dat, negative_dat), (anchor_loc, positive_loc, negative_loc)

# weighted loss based on imbalanced data
def get_loss_weight(img_num_list):
    # number of awned:   123571
    # number of awnless: 5236
    # set the reduction ratio for awned sample to 0.03
    
    weight_list = 1/img_num_list.float()
    normalized_weight = weight_list * len(weight_list)/weight_list.sum()
    return normalized_weight