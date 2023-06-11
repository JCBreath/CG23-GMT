# pyTorch=1.9.1+cu111

import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import torch.optim as optim
import time

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models import GMT, GMT_D
from utils import Dataset, Logger, fod, ssim3d

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def change_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad

def inference(model, args, dataset):
    logger = Logger(args.log_path)
    logger.add(['Timestep', 'PSNR'])

    src_idx = args.source
    label_idx = args.target
    c = dataset.c
    one = np.ones((3,c[0],c[1],c[2]))

    for xx in range(0,c[0]):
        print("Creating weight mapping: {}/{}".format(xx,c[0]),end='\r')
        for yy in range(0,c[1]):
            for zz in range(0,c[2]):
                dx = abs(xx+0.5-c[0]/2)
                dy = abs(yy+0.5-c[1]/2)
                dz = abs(zz+0.5-c[2]/2)
                one[0,xx,:,:] = 1-dx/(c[0]/2)
                one[1,:,yy,:] = 1-dy/(c[1]/2)
                one[2,:,:,zz] = 1-dz/(c[2]/2)

    one = one[0] * one[1] * one[2]
    # one = np.fromfile('{}weight_map_{}.iw'.format(args.pth_path,c[0]),dtype='<f')
    # one = np.fromfile('../GMT_INFERENCE/weight_map_{}.iw'.format(c[0]),dtype='<f')
    one = one.reshape((c[0],c[1],c[2])).transpose()
    one = torch.FloatTensor(one).cuda()


    for dataset_start in range(1, 151, 5):
        dataset.start = dataset_start
        dataset.num_samples = 5

        dataset.LoadData()
        
        for i in range(dataset.num_samples):
            s = torch.FloatTensor(dataset.data[src_idx][i])
            s = s.view(1,1,s.size(0),s.size(1),s.size(2))
            s = s.cuda()

            result = np.zeros((dataset.dim[0],dataset.dim[1],dataset.dim[2]))
            weight = np.zeros((dataset.dim[0],dataset.dim[1],dataset.dim[2]))

            result = torch.FloatTensor(result).cuda()
            weight = torch.FloatTensor(weight).cuda()

            step_size = 32
            # step_size = 48

            count = 1

            label = dataset.labels[label_idx]
            label = label.view(1,1,len(label)).cuda()

            print(label)

            for xx in range(0,dataset.dim[0],step_size):
                for yy in range(0,dataset.dim[1],step_size):
                    for zz in range(0,dataset.dim[2],step_size):
                        
                        x = min(xx, dataset.dim[0]-dataset.c[0])
                        y = min(yy, dataset.dim[1]-dataset.c[1])
                        z = min(zz, dataset.dim[2]-dataset.c[2])

                        c = s[0,0,x:x+dataset.c[0],y:y+dataset.c[1],z:z+dataset.c[2]].view(1,1,dataset.c[0],dataset.c[1],dataset.c[2])
                        
                        with torch.no_grad():
                            v = model(c, label)

                        v = v[0][0].detach()

                        result[x:x+dataset.c[0],y:y+dataset.c[1],z:z+dataset.c[2]] += v * one
                        weight[x:x+dataset.c[0],y:y+dataset.c[1],z:z+dataset.c[2]] += one

                        print(count, end='\r')
                        count += 1

            result = torch.divide(result, weight).cpu().numpy()
            data = np.asarray(result,dtype='<f')

            p = peak_signal_noise_ratio(dataset.data[label_idx][i],data)

            print("Timestep: {}, PSNR: {}".format(i+dataset.start,p))
            logger.add([i+dataset.start,p])
            
            data = data.reshape(dataset.dim[0],dataset.dim[1],dataset.dim[2]).transpose()

            data = data.flatten()
            data = (data+1)/2
            data = np.clip(data, 0, 1)
            filename = '{}{}-{:04d}'.format(args.out_path, args.dataset,i+dataset.start)+'.iw'
            data.tofile(filename,format='<f')

def generate_weight_map(c):
    one = np.ones((3,c[0],c[1],c[2]), dtype='<f')

    for xx in range(0,c[0]):
        print("Creating weight mapping: {}/{}".format(xx,c[0]),end='\r')
        for yy in range(0,c[1]):
            for zz in range(0,c[2]):

                dx = abs(xx+0.5-c[0]/2)
                dy = abs(yy+0.5-c[1]/2)
                dz = abs(zz+0.5-c[2]/2)
                one[0,xx,:,:] = 1-dx/(c[0]/2)
                one[1,:,yy,:] = 1-dy/(c[1]/2)
                one[2,:,:,zz] = 1-dz/(c[2]/2)

    one = one[0] * one[1] * one[2]
    one = one.transpose().flatten()
    one.tofile('weight_map_{}.iw'.format(c[0]),format='<f')


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "GMT"')
    parser.add_argument('--epoch', type=int, default=4000)
    parser.add_argument('--max_epoch', type=int, default=4000)
    parser.add_argument('--data_path', type=str, default='/mnt/g')
    parser.add_argument('--model_path', type=str, default='/gmt_models/inference')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='tangaroa')
    parser.add_argument('--crop', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=40)
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--source', type=int, default=0)
    parser.add_argument('--model', type=str, default='model')

    args = parser.parse_args()

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
            # 'crop' : (96,96,96),
            # 'crop' : (128,128,128),
            'start' : 46,
            'num' : 60,
        },
        'half-cylinder':{
            'vars':['velocity', 'vorticity', 'divergence_magnitude', 'acceleration'],
            'dim' : (640,240,80),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        },
        'ionization':{
            'vars':['H', 'H+', 'He', 'He+'],
            'dim' : (600,248,248),
            'crop' : (64,64,64),
            'start' : 31,
            'num' : 40,
        },
        'combustion':{
            'vars':['MF', 'CHI','YOH','HR'],
            'dim' : (480,720,120),
            'crop' : (64,64,64),
            # 'crop' : (96,96,96),
            'start' : 31,
            'num' : 40,
        }
    }

    param = args.params[args.dataset]
    
    print(param)

    model_path = args.model_path

    args.data_path = args.data_path
    args.log_path = model_path + 'logs/'
    args.pth_path = model_path
    args.out_path = model_path + param['vars'][args.target] + '/'
    
    check_path(args.model_path)
    check_path(args.log_path)
    check_path(args.pth_path)
    check_path(args.out_path)

    print(args)

    c_dim = len(param['vars'])

    model = GMT(c_dim=c_dim)
    model_d = GMT_D()
    
    model.load_state_dict(torch.load('{}{}.pth'.format(args.model_path, args.epoch)))

    model.cuda()
    model_d.cuda()

    dataset = Dataset(args, param)

    inference(model, args, dataset)
