import os, re
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import logger as logger_
from datetime import datetime

class Trainer:
    def __init__(self, model, dataloader, loss_func, optimizer, scheduler, n_epochs, acc_func=None, train_name='default', 
                 resume_dict=None, ckp_dir=None, resume_ep='latest',
                 logger='tensorboard', log_dir=None, log_step_interval=100, log_graph=False, log_hparam={}, comment=None):
        '''
        If resume_dict is not None then load from resume_dict and ignore files stored in ckp_dir
        '''
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.acc_func = acc_func
        self.logger = logger
        self.log_hparam = log_hparam
        self.ckp_dir = ckp_dir
        self.log_dir = log_dir
        self.log_step_interval = log_step_interval
        self.start_epoch=0
        self.epoch_loss = None
        self.epoch_acc = None
        self.epoch_val_acc = None
        self.epoch_num = None
        self.step_loss = None
        self.step_acc = None
        self.step_num = None
        self.steps_per_epoch = len(self.dataloader)
        if self.ckp_dir is None:
            self.ckp_dir = './checkpoints/{}/{}'.format(train_name, self.timestamp)
        if not os.path.isdir(self.ckp_dir):
            os.makedirs(self.ckp_dir)
        # find model
        ckp_dir_file_names = os.listdir(self.ckp_dir)
        if model is None:
            if 'jit_model.pth' in ckp_dir_file_names:        
                self.model = torch.jit.load(os.path.join(self.ckp_dir, 'jit_model.pth'))
            else:
                raise ValueError('No model found')
        # find resume ep
        if resume_ep == 'latest':
            epoch_file_list = glob(os.path.join(self.ckp_dir, 'epoch_*.pth'))
            if len(epoch_file_list):
                prog = re.compile("epoch_([0-9]+).pth")
                epoch_number_list = [int(prog.findall(i)[0]) for i in epoch_file_list]
                resume_ep = max(epoch_number_list)
                print(f'Latest epoch {resume_ep} found.')
            else:
                resume_ep = None
        else:
            assert resume_ep.isnumeric(), ValueError("resume_ep must be numeric or 'latest'.")
            assert os.path.isfile(os.path.join(self.ckp_dir, f"epoch_{resume_ep}.pth")), \
                   FileNotFoundError(f"epoch {resume_ep} not found in {self.ckp_dir}")
            resume_ep = int(resume_ep)
        if resume_dict is not None:
            self.load_epoch(resume_dict)
        elif resume_ep is not None:
            self.load_epoch(os.path.join(self.ckp_dir, f"epoch_{resume_ep}.pth"))
        # Logger
        if self.logger == 'tensorboard':
            if self.log_dir is None:
                self.log_dir = 'tb_logs/{}'.format(train_name)
                self.log_dir = os.path.join(self.log_dir, self.timestamp)
            self.logger = logger_.TensorBoardLogger(self.log_dir)
        elif self.logger == 'neptune':
            self.logger = logger_.NeptuneLogger(train_name, self.log_dir, hparams=log_hparam, comment=comment)
        elif isinstance(self.logger, logger_._Logger):
            pass 
        else:
            raise ValueError(f'Unknown type of logger {self.logger}')
        self.record_params = ['loss_func', 'step_loss', 'step_acc']
        self.epoch_finish_hook = []
        self.logger.log_init(self)

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            self.epoch_num = epoch
            current_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            print('---- start epoch: {}/{}\tlearning rate:{:.2E} ----'.format(self.epoch_num, self.n_epochs, current_lr))
            self.epoch_loss, self.epoch_acc = self.train_epoch()
            for fn in self.epoch_finish_hook:
                fn(self)
            # tb_writer.add_scalar('epoch/lr', scheduler.get_lr()[0], epoch)
            # tb_writer.add_scalar('epoch/train_loss',epoch_loss, epoch)
            # if epoch_acc is not None:
            #     tb_writer.add_scalar('epoch/train_acc', epoch_acc, epoch)
            # tb_writer.flush()
            self.scheduler.step()
            print('end epoch: {}/{}\ttrain loss: {:.2f}\ttrain acc: {}\n'.format(self.epoch_num, self.n_epochs, self.epoch_loss, self.epoch_acc))
            self.save_epoch()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        tbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for batch_idx, (data, target) in tbar:
            self.step_num = batch_idx
            self.optimizer.zero_grad()
            if type(data) is tuple:
                data = tuple(d.cuda() for d in data)
                model_output = [self.model(d) for d in data]
            elif type(data) is torch.Tensor:
                model_output = self.model(data.cuda())
            else:
                model_output = self.model(data)
                # raise TypeError(f'Unknown type of input data{type(data)}')
            loss = self.loss_func(model_output, target)
            self.step_loss = loss.item()
            total_loss += self.step_loss
            loss.backward()
            self.optimizer.step()
            self.logger.step(self)
            if self.acc_func is None:
                tbar.set_description('loss: {:.2f}'.format(self.step_loss))
            else:
                acc = self.acc_func(model_output, target)
                self.step_acc = acc
                total_acc += self.step_acc
                tbar.set_description('loss: {:.2f}, acc: {:.2f}'.format(self.step_loss, acc))
        epoch_loss = total_loss/len(self.dataloader)
        epoch_acc = None
        if self.acc_func is not None:
            epoch_acc = total_acc/len(self.dataloader)
        return epoch_loss, epoch_acc

    def add_epoch_hook(self, func):
        self.epoch_finish_hook.append(func)
        return len(self.epoch_finish_hook) - 1
    
    def remove_epoch_hook(self, i):
        self.epoch_finish_hook.pop(i)
        
    def save_jit_model(self):
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, os.path.join(self.ckp_dir, 'jit_model.pth'))

    def save_epoch(self):
        save_path = save_path = os.path.join(self.ckp_dir, 'epoch_{}.pth'.format(self.epoch_num))
        torch.save({'epoch': self.epoch_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()},
        save_path)

    def load_epoch(self, resume_dict):
        if type(resume_dict) is str:
            resume_dict = torch.load(resume_dict)
        self.start_epoch = resume_dict['epoch'] + 1
        self.model.load_state_dict(resume_dict['model_state_dict'])
        self.optimizer.load_state_dict(resume_dict['optimizer'])
        self.scheduler.load_state_dict(resume_dict['scheduler'])


# backward capability
def train(model, dataloader, loss_func, optimizer, scheduler, n_epochs, resume_dict=None, ckp_dir=None, 
          tb_log_dir=None, tb_log_step_interval=100):
    trainer = Trainer(model, dataloader, loss_func, optimizer, scheduler, n_epochs, resume_dict=resume_dict, 
                      ckp_dir=ckp_dir, tb_log_dir = tb_log_dir, log_step_interval=tb_log_step_interval)
    trainer.train()


def train_deprecated(model, dataloader, loss_func, optimizer, scheduler, n_epochs, resume_dict=None, ckp_dir=None, 
          tb_log_dir=None, tb_log_step_interval=100):
    if tb_log_dir is None:
        tb_log_dir = 'tb_logs/{}'.format(os.path.split(ckp_dir)[-1])
    tb_writer = SummaryWriter(tb_log_dir)
    start_epoch = 0
    if resume_dict is not None:
        start_epoch = resume_dict['epoch']
        model.load_state_dict(resume_dict['model_state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        scheduler.load_state_dict(resume_dict['scheduler'])
    if ckp_dir is None:
        ckp_dir = './checkpoints/default'
    if not os.path.isdir(ckp_dir):
        os.makedirs(ckp_dir)
    for epoch in range(start_epoch, n_epochs):
        current_lr = [group['lr'] for group in optimizer.param_groups][0]
        print('start epoch: {}/{}\tlearning rate:{:.2E}\t'.format(epoch, n_epochs, current_lr))
        epoch_loss, epoch_acc = train_epoch(model, dataloader, loss_func, optimizer, epoch, tb_writer=tb_writer, tb_log_step_interval=tb_log_step_interval)
        tb_writer.add_scalar('epoch/lr', current_lr, epoch)
        tb_writer.add_scalar('epoch/train_loss',epoch_loss, epoch)
        if epoch_acc is not None:
            tb_writer.add_scalar('epoch/train_acc', epoch_acc, epoch)
        tb_writer.flush()
        scheduler.step()
        print('end epoch: {}/{}\ttrain loss: {:.2f}\ttrain acc: {}'.format(epoch, n_epochs, epoch_loss, epoch_acc))
        print('-------\n')
        ckp_path = os.path.join(ckp_dir, 'epoch_{}.pth'.format(epoch+1))
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': epoch_loss,
                    'acc': epoch_acc},
                   ckp_path)


def train_epoch(model, dataloader, loss_func, optimizer, current_epoch, acc_func=None, tb_writer=None, tb_log_step_interval=100):
    model.train()
    total_loss = 0
    total_acc = 0
    tb_logger_total_loss = 0
    tb_logger_total_acc = 0
    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (data, target) in tbar:
        data = tuple(d.cuda() for d in data)
        optimizer.zero_grad()
        model_output = [model(d) for d in data]
        loss = loss_func(model_output, target)
        total_loss += loss.item()
        tb_logger_total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if acc_func is None:
            tbar.set_description('loss: {:.2f}'.format(loss.item()))
        else:
            acc = acc_func(model_output, target)
            total_acc += acc
            tb_logger_total_acc += acc
            tbar.set_description('loss: {:.2f}, acc: {:.2f}'.format(loss.item(), acc))
        # Tensorboar logging
        if tb_writer is not None:
            if batch_idx % tb_log_step_interval == tb_log_step_interval - 1:
                tb_writer.add_scalar('step/train_loss', tb_logger_total_loss/tb_log_step_interval, current_epoch*len(dataloader)+batch_idx)
                tb_logger_total_loss = 0
                if acc_func is not None:
                    tb_writer.add_scalar('step/train_acc', tb_logger_total_acc/tb_log_step_interval, current_epoch*len(dataloader)+batch_idx)
                    tb_logger_total_acc = 0
            tb_writer.flush()
    epoch_loss = total_loss/len(dataloader)
    epoch_acc = None
    if acc_func is not None:
        epoch_acc = total_acc/len(dataloader)
    return epoch_loss, epoch_acc


"""if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    resume_dict = None
    # resume_dict = torch.load('./checkpoints/epoch_30.pth')
    print('Training on large dataset')
    lr = 1e-3
    batch_size = 16
    print('Load dataset')
    image_transform = transforms.Compose([transforms.CenterCrop((700, 1000)),
                                          transforms.RandomCrop((700, 700)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    dataset = LimeCardinalTripletDataset('./terraref/scanner3DTop/lime_cardinal_dataset/dataset/', transform=image_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    print('Init model')
    model = resnet_50_embedding()
    loss_func = TripletMarginLoss()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    print('Train 1 epoch')
    train(model, dataloader, loss_func, optimizer, scheduler, 64, resume_dict=resume_dict)
"""
