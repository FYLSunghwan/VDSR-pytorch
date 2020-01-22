from settings import Settings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainDataset, TestDataset
from torchvision.transforms import *
from torch.autograd import Variable
from model import Net
from tensorboardX import SummaryWriter
from PIL import Image
import math
import numpy as np

settings = Settings()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(settings.dataset_info['291'])
print(settings.lr)


# Here is the function for PSNR calculation
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1 / rmse)


def load_dataset(dataset='train'):
    if settings.num_channels == 1:
        is_gray = True
    else:
        is_gray = False

    if dataset == 'train':
        print('Loading train datasets...')
        train_set = TrainDataset(settings=settings)
        return DataLoader(dataset=train_set, num_workers=settings.num_threads, batch_size=settings.batch_size,
                          shuffle=True)
    elif dataset == 'test':
        print('Loading test datasets...')
        test_set = TestDataset(settings=settings)
        return DataLoader(dataset=test_set, num_workers=settings.num_threads, batch_size=settings.test_batch_size,
                          shuffle=False)
    elif dataset == 'gdata':
        print('github')
        train_set = DatasetFromHdf5('datasets/data/train.h5')
        return DataLoader(dataset=train_set, num_workers=settings.num_threads, batch_size=settings.test_batch_size,
                          shuffle=True)
      
        
def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    
    
def train(training_data_loader, testing_data_loader, optimizer, model, criterion, epoch, writer, log_iter, test_log_iter):
    #lr = adjust_learning_rate(optimizer, epoch)
    #lr = 1e-4
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]
        
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    avg_loss = 0.0
    for _ in range(100):
        for iteration, batch in enumerate(training_data_loader):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input), target)
            loss.backward() 

            #nn.utils.clip_grad_norm_(model.parameters(), settings.clip) 
            optimizer.step()

            writer.add_scalar('loss/L2_loss', loss.item()/settings.batch_size, log_iter)
            writer.add_scalar('loss/PSNR', 10*math.log10(1/loss.item()),log_iter)
            
            avg_loss += loss.item()
            log_iter += 1
        
        with torch.no_grad():
            avg_psnr = 0.0
            cnt = 0
            for iteration, batch in enumerate(testing_data_loader):
                bicubic, hires = Variable(batch[0], requires_grad=False), batch[1]
                bicubic = bicubic.to(device)
                out = model(bicubic).cpu().detach().numpy().squeeze(0)
                hires = hires.numpy().squeeze(0)
                avg_psnr += PSNR(hires, out)
                cnt += 1
            avg_psnr /= cnt
            writer.add_scalar('loss/PSNR_test', 10*math.log10(1/loss.item()),test_log_iter)
            test_log_iter += 1
        
        
    return log_iter, test_log_iter


train_data = load_dataset('train')
test_data = load_dataset('test')
print(train_data)
print(test_data)


model = Net().to(device)
writer = SummaryWriter()

optimizer = optim.SGD(model.parameters(), lr=settings.lr, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.MSELoss()

total_epoch = 80
total_iter = 0
test_iter = 0

for epoch in range(total_epoch):
    total_iter, test_iter = train(train_data, test_data, optimizer, model, criterion, epoch, writer=writer, log_iter=total_iter, test_log_iter=test_iter)
    scheduler.step()
    save_checkpoint(model, epoch)