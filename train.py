# pyTorch=1.9.1+cu111

import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import torch.optim as optim
import time

from models import GMT, GMT_D
from utils import Dataset, Logger, fod, ssim3d

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def change_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad

def require_g_grad(model, model_d):
    change_grad(model, True)
    change_grad(model_d, False)

def require_d_grad(model, model_d):
    change_grad(model, False)
    change_grad(model_d, True)


def train(model, model_d, args, dataset):
    logger = Logger(args.log_path)
    logger.add(['Epoch', 'MSE'])

    dataset.LoadData()

    model_optim = optim.Adam(model.parameters(), lr=1e-4,betas=(0.9,0.999))
    if model_d is not None:
        d_optim = optim.Adam(model_d.parameters(), lr=1e-4,betas=(0.9,0.999))

    mse = nn.MSELoss()

    for epoch in range(args.epoch+1, args.max_epoch+1):
        start_time = time.time()
        
        train_loader = dataset.PrepareDataLoader()
        
        
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_adv_loss = 0
        epoch_feat_loss = 0
        epoch_d_loss = 0
        
        count = 1

        for batch_idx, (src, gt, label) in enumerate(train_loader):
            src = src.cuda()
            gt = gt.cuda()
            label = label.cuda()

            require_d_grad(model, model_d)
            d_optim.zero_grad()

            score, _ = model_d(gt)
            loss_real = (torch.mean(score-1)) ** 2     
            score, _ = model_d(model(src, label))
            loss_fake = (torch.mean(score-0)) ** 2

            loss_d = loss_real + loss_fake
            loss_d.backward()
            d_optim.step()
            epoch_d_loss += loss_d.item()
            
            # dis_time = 1

            dis_time = 5
            if epoch > 100:
                dis_time = 4
            if epoch > 200:
                dis_time = 3
            if epoch > 300:
                dis_time = 2
            if epoch > 400:
                dis_time = 1

            if count % dis_time == 0:
                require_g_grad(model, model_d)
                model_optim.zero_grad()
                pred = model(src, label)
                score, feat_pred = model_d(pred)
                _, feat_gt = model_d(gt)

                grad_loss = mse(fod(pred,1), fod(gt,1)) + mse(fod(pred,2), fod(gt,2)) + mse(fod(pred,3), fod(gt,3))
                dssim_loss = torch.clamp(1.0-ssim3d(pred,gt), min=0.0)
                mse_loss = mse(pred,gt)
                adv_loss = (torch.mean(score-1)) ** 2
                feat_loss = 0
                for i in range(len(feat_gt)):
                    feat_loss += mse(feat_pred[i], feat_gt[i])
                epoch_mse_loss += mse_loss.item()
                epoch_adv_loss += adv_loss.item()
                epoch_feat_loss += feat_loss.item()
                loss = mse_loss + 0.1*dssim_loss  + 2.0*grad_loss + 1e-2*feat_loss + 1e-3*adv_loss
                # loss = mse_loss

                loss.backward()
                model_optim.step()
                time_elapsed = time.time() - start_time
                print("Epoch [{}], Iter [{}/{}], Loss [{:.4f}], Time Elapsed [{:.2f}], pred_std [{:.4f}], gt_std [{:.4f}]".format(epoch, count, len(train_loader.dataset), loss.mean().item(), time_elapsed, torch.std(pred), torch.std(gt)), end='\r')
                
            count += 1

        time_elapsed = time.time() - start_time
        logger.add([epoch, epoch_loss])
        print("Epoch [{}], MSE [{:.4f}], FEAT [{:.2f}], ADV [{:.2f}], D [{:.2f}], Time [{:.2f}]]".format(epoch, epoch_mse_loss, epoch_feat_loss, epoch_adv_loss, epoch_d_loss, time_elapsed))

        if epoch%100 == 0:
            print("Model saved at Epoch [{}]".format(epoch))
            torch.save(model.state_dict(),'{}{}.pth'.format(args.pth_path, epoch))
            torch.save(model_d.state_dict(),'{}{}_d.pth'.format(args.pth_path, epoch))


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "GMT"')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=4000)
    parser.add_argument('--path', type=str, default='/mnt/g')
    parser.add_argument('--model_path', type=str, default='/gmt_models')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='tangaroa')
    parser.add_argument('--crop', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=40)
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--source', type=int, default=0)

    args = parser.parse_args()


    args.data_path = args.path + '/vis_data/'
    args.log_path = args.path + args.model_path + '/logs/'
    args.pth_path = args.path + args.model_path + '/pths/'
    args.out_path = args.path + args.model_path + '/outs/'
    
    check_path(args.path + args.model_path)
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
            'vars':['H', 'H+'],
            'dim' : (600,248,248),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        },
        'half-cylinder':{
            'vars':['velocity', 'vorticity', 'divergence_magnitude', 'acceleration'],
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
    model_d = GMT_D()
    
    model.load_state_dict(torch.load('{}pre-train_{}.pth'.format(args.pth_path, 500)))

    model.cuda()
    model_d.cuda()

    dataset = Dataset(args, param)


    train(model, model_d, args, dataset)
