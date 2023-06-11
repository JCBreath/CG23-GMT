# pyTorch=1.9.1+cu111

import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import torch.optim as optim
import time

from models import GMT
from utils import Dataset, Logger

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def change_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad


def train(model, args, dataset):
    logger = Logger(args.log_path)
    logger.add(['Epoch', 'MSE'])

    dataset.LoadData()
    model_optim = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.999))
    mse = nn.MSELoss()
    change_grad(model, True)
    

    for epoch in range(args.epoch+1, args.max_epoch+1):
        start_time = time.time()
        
        train_loader = dataset.PrepareDataLoader()
        
        
        epoch_loss = 0
        epoch_mse_loss = 0
        
        count = 1

        for batch_idx, (src, gt, label) in enumerate(train_loader):

            src = src.cuda()
            gt = gt.cuda()
            label = label.cuda()

            model_optim.zero_grad()
            pred = model(gt,label)

            if pred.isnan().any():
                torch.save(model.state_dict(),'{}pre-train_{}.pth'.format(args.pth_path, epoch))
                exit()

            mse_loss = mse(pred,gt)

            if mse_loss.isnan().any():
                torch.save(model.state_dict(),'{}pre-train_{}.pth'.format(args.pth_path, epoch))
                exit()

            loss = mse_loss
            loss.backward()
            model_optim.step()
            epoch_mse_loss += mse_loss.item()
            time_elapsed = time.time() - start_time

            print("Epoch [{}], Iter [{}/{}], Loss [{:.4f}], Time Elapsed [{:.2f}]".format(epoch, count, len(train_loader.dataset), loss.mean().item(), time_elapsed), end='\r')
                
            count += 1

        time_elapsed = time.time() - start_time
        logger.add([epoch, epoch_mse_loss])
        print("Epoch [{}], MSE [{:.4f}], Time [{:.2f}]]".format(epoch, epoch_mse_loss, time_elapsed))

        if epoch%100 == 0:
            print("Model saved at Epoch [{}]".format(epoch))
            torch.save(model.state_dict(),'{}pre-train_{}.pth'.format(args.pth_path, epoch))

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "GMT"')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--data_path', type=str, default='/mnt/f')
    parser.add_argument('--model_path', type=str, default='/gmt_models')
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--dataset', type=str, default='tangaroa')
    parser.add_argument('--crop', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=40)

    args = parser.parse_args()

    args.data_path = args.data_path
    args.log_path = args.model_path + '/logs/'
    args.pth_path = args.model_path + '/pths/'
    args.out_path = args.model_path + '/outs/'
    
    check_path(args.model_path)
    check_path(args.log_path)
    check_path(args.pth_path)
    check_path(args.out_path)

    print(args)

    args.params = {
        'tangaroa':{
            'vars':['velocity','vorticity','divergence','acceleration'],
            'dim' : (300,180,120),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        },
        'mantle':{
            'vars':['conductivity_anomaly','temperature_anomaly','density_anomaly','expansivity_anomaly'],
            'dim' : (360,201,180),
            'crop' : (64,64,64),
            'start' : 46,
            'num' : 60,
        },
        'ionization':{
            'vars':['H', 'H+','He','He+'],
            'dim' : (600,248,248),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        },
        'half-cylinder':{
            'vars':['6400', '160'],
            'dim' : (640,240,80),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        },
        'combustion':{
            'vars':['MF','CHI','YOH', 'HR'],
            'dim' : (480,720,120),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        }
    }

    param = args.params[args.dataset]
    
    print(param)

    c_dim = len(param['vars'])

    model = GMT(c_dim=c_dim)
    model.cuda()

    dataset = Dataset(args, param)

    train(model, args, dataset)