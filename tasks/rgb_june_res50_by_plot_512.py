import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import date
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils import data
import torch.multiprocessing
from torchvision import transforms, models
import numpy as np
from pytorch_metric_learning import losses as pml_losses
from dataset.dataset import OPENStereoDataset, open_split_train_validate
from train import Trainer 
import util_hooks
from hparam import HParam
torch.multiprocessing.set_sharing_strategy('file_system')

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, time_range, weight = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', class_by='scan_date') -> None:
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.class_by = class_by
        self.time_range = time_range

    def forward(self, input, target):
        return super().forward(input, target[self.class_by].cuda())

def log_epoch_acc_hook(trainer):
    trainer.logger.exp.log_metric('epoch/val_acc', y=trainer.epoch_val_acc ,x=trainer.logger.epoch_counter)

def acc_func(vectors, labels):
    # TODO change all to torch instead of numpy
    if type(vectors) == torch.Tensor:
        device = vectors.device
    else:
        device = None
    predict_cls = torch.argmax(F.softmax(vectors, dim=1), dim=1).type(labels[dataset_class_by].dtype)
    acc = torch.mean(labels[dataset_class_by].to(device).eq(predict_cls).type(torch.float))
    return acc  

exp_name = os.path.splitext(os.path.basename(__file__))[0]
resume_dict = None

hps = HParam(img_crop_size = 512,
             img_resize = 512,
             resnet_n = 50,
             init_lr = 1e-2,
             batch_size = 30,
             time_ebd_linear=True,
             time_head_fix = True,
             time_mask='linear',
             num_time_ebd_slot = 1,
             sampler_date_shuffle=True)

print('Experiment name: {}'.format(exp_name))
print('Load dataset')
rgb_mean, rgb_sd = (103.44197739/255.,108.46646883/255.,82.57383395/255.), (0.39088993, 0.38828301, 0.44437335)
image_transform = transforms.Compose([transforms.RandomCrop(hps.img_crop_size, pad_if_needed=True),
                                      # transforms.Resize(hps.img_resize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(rgb_mean, rgb_sd)])
val_image_transform = transforms.Compose([transforms.RandomCrop(hps.img_crop_size, pad_if_needed=True),
                                      # transforms.Resize(hps.img_resize),
                                      transforms.ToTensor(),
                                      transforms.Normalize(rgb_mean, rgb_sd)])
exclude_cultivar = ['Big_Kahuna', 'SP1615']
start_date = date(2017, 6, 1)
end_date = date(2017, 6, 30)
# dataset = PosetPairDataset(OPENScanner3dDataset('./datasets/OPEN', start_date=start_date, end_date=end_date, exclude_date=exclude_date,
#                                                       transform=image_transform), select_by_label='scan_date')
with open('./dataset_split/val_cultivars.txt', 'r') as f:
    val_labels = set([l.strip() for l in f.readlines()])
dataset = OPENStereoDataset('./datasets/OPEN',
                            start_date=start_date, end_date=end_date,
                            exclude_cultivar=exclude_cultivar,
                            image_file_suffix='.png',
                            transform=image_transform)
dataset_class_by = 'plot_cls'

time_range = [min(dataset.labels['scan_date']), max(dataset.labels['scan_date'])]
print(f"date start from {date.fromordinal(time_range[0])} to {date.fromordinal(time_range[1])}")
num_class = len(np.unique(dataset.labels[dataset_class_by]))
_, val_dataset = open_split_train_validate(dataset, val_labels=val_labels, class_by='cultivar')
train_dataset = dataset
val_dataset.transform = val_image_transform
# train_dataset = LabelTrainDataset(train_dataset, 'scan_date')
# val_dataset = LabelTrainDataset(val_dataset, 'scan_date')
# batch_sampler = TripletPosetNearDateBatchSampler(train_dataset, class_by='plot', date_by='scan_date', 
#                                                  batch_size=hps.batch_size, class_size=hps.class_size,
#                                                  neighbor_distance=hps.neighbor_distance, shuffle_date=hps.sampler_date_shuffle)
dataloader = data.DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=True, num_workers=22)
val_dataloader = data.DataLoader(val_dataset, batch_size=125, shuffle=False, num_workers=22)
print('Init model')
class SelectImg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x[0].cuda()

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_class)

triplet_loss_func = pml_losses.TripletMarginLoss()
loss_func = CrossEntropyLoss(time_range=time_range, class_by=dataset_class_by)
model.cuda()
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=hps.init_lr)
scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
t = Trainer(model, dataloader, loss_func, optimizer, scheduler, n_epochs=120, train_name=exp_name,
            acc_func=acc_func,
            logger='neptune', log_dir='zeyu/reverse-pheno', log_hparam=hps.__dict__)


hooks = [util_hooks.construct_test_hook(val_dataloader),
         util_hooks.construct_metric_hook(acc_func),
         log_epoch_acc_hook]
for h in hooks:
    t.add_epoch_hook(h)
 
t.train()