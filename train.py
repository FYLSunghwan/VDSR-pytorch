from settings import Settings
import matplotlib.pyplot as plt

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainDataset, TestDataset
from torchvision.transforms import *
from torch.autograd import Variable
from model import Net
from model import SRCNN
from tensorboardX import SummaryWriter
from PIL import Image
from utils.psnr import PSNR
import numpy as np
import math


def load_dataset(dataset, settings):
    if settings.num_channels == 1:
        is_gray = True
    else:
        is_gray = False

    if dataset == 'train':
        print('Loading train datasets...')
        train_set = TrainDataset(settings=settings)
        return DataLoader(dataset=train_set, num_workers=settings.num_threads, batch_size=settings.batch_size,
                          shuffle=True, drop_last=True)
    elif dataset == 'test':
        print('Loading test datasets...')
        test_set = TestDataset(settings=settings)
        return DataLoader(dataset=test_set, num_workers=settings.num_threads, batch_size=settings.test_batch_size,
                          shuffle=False)
        
        
def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    
    
def train(training_data_loader, testing_data_loader, device, settings, optimizer, model, criterion, epoch, writer, log_iter, test_log_iter):
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]
        
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    avg_loss = 0.0
    for i in range(2490):
        if i%100==0:
            print(">> {}/2490".format(i))
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
            psnrs = []
            bic_psnrs = []
            cnt = 0
            for iteration, batch in enumerate(testing_data_loader):
                bicubic, hires = Variable(batch[0], requires_grad=False), batch[1]
                bicubic = bicubic.to(device)
                out = model(bicubic).cpu().detach().numpy().squeeze(0)
                
                bicubic = bicubic.cpu().detach().numpy().squeeze(0)
                hires = hires.squeeze(0)
                
                bicubic *= 255
                hires *= 255
                out *= 255
                
                bicubic = np.float32(bicubic).transpose(2,1,0)
                hires = np.float32(hires).transpose(2,1,0)
                out = np.float32(out).transpose(2,1,0)

                ps = PSNR(hires, out, ycbcr=True)
                bps = PSNR(hires, bicubic, ycbcr=True)

                psnrs.append(ps)
                bic_psnrs.append(bps)
                
            avg_psnr = np.mean(psnrs)
            avg_bic_psnr = np.mean(bic_psnrs)
            writer.add_scalar('loss/PSNR_test', avg_psnr,test_log_iter)
            writer.add_scalar('loss/PSNR_test_improve', avg_psnr-avg_bic_psnr, test_log_iter)
            test_log_iter += 1
        
    return log_iter, test_log_iter
        
def main():
    settings = Settings()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(settings.dataset_info['291'])
    print(settings.lr)

    train_data = load_dataset('train', settings)
    test_data = load_dataset('test', settings)
    
    #model = Net().to(device)
    model = torch.load('checkpoint/model_epoch_2.pth')['model'].to(device)
    writer = SummaryWriter()
    
    #optimizer = optim.SGD(model.parameters(), lr=settings.lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=settings.lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = nn.MSELoss()

    total_epoch = 80
    total_iter = 0
    test_iter = 0
    
    for epoch in range(total_epoch):
        total_iter, test_iter = train(train_data, test_data, device, settings, optimizer, model, criterion, epoch, writer=writer, log_iter=total_iter, test_log_iter=test_iter)
        #scheduler.step()
        save_checkpoint(model, epoch)
        
    dbs = []
    for _, batch in enumerate(test_data):
        bicubic, hires = Variable(batch[0], requires_grad=False), batch[1]
        bicubic = bicubic.to(device)
        out = model(bicubic).cpu().detach().numpy().squeeze(0)

        bicubic = bicubic.cpu().detach().numpy().squeeze(0)
        hires = hires.squeeze(0)

        hires *= 255
        out *= 255

        bicubic = np.uint8(bicubic).transpose(2,1,0)
        hires = np.float32(hires).transpose(2,1,0)
        out = np.float32(out).transpose(2,1,0)

        ps = PSNR(hires, out, ycbcr=True)
        dbs.append(ps)

    print('total PSNR:', np.mean(dbs))
    
if __name__ == '__main__':
    main()