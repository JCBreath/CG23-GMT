import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import datetime, csv

class Logger():
    def __init__(self,log_path):
        now = datetime.datetime.now()
        self.log_file = open('{}log-{}.csv'.format(log_path,now.strftime("%m-%d-%Y-%H-%M")), 'w', newline='')
        self.log_writer = csv.writer(self.log_file)

    def add(self, row):
        self.log_writer.writerow(row)

class Dataset():
    def __init__(self,args,param):
        self.args = args
        self.pretrain = self.args.mode == 'pretrain'
        self.dim = param['dim']
        self.start = param['start']
        self.num_samples = param['num']
        self.c = param['crop']
        self.data_path = "{}{}/".format(args.data_path, args.dataset)
        self.variables = param['vars']
        num_target = len(self.variables)
        labels = np.zeros((num_target, num_target))
        labels[np.arange(num_target), np.arange(num_target)] = 1
        self.labels = torch.FloatTensor(labels)
        self.data = []
    def LoadData(self):
        self.data = []

        for i in range(len(self.variables)):
            v_data = []
            for j in range(self.num_samples):
                if self.args.dataset == 'ionization':
                    v = np.fromfile('{}{}-{:04d}.iw'.format(self.data_path, self.variables[i], j+self.start),dtype='<f')
                elif self.args.dataset == 'half-cylinder':
                    v = np.fromfile('{}{}/{}-{}-magnitude-{:04d}.iw'.format(self.data_path, self.variables[i], self.args.dataset, self.variables[i], j+self.start),dtype='<f')
                else:
                    v = np.fromfile('{}{}/{}-{:04d}.iw'.format(self.data_path, self.variables[i], self.args.dataset, j+self.start),dtype='<f')
                v = v.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
                v = 2.0*v-1.0
                v = np.asarray(v)
                v_data.append(v)
                print("Loading {} [{}/{}]".format(self.variables[i], j, self.num_samples), end='\r')
            print("Loading {} [{}/{}]".format(self.variables[i], self.num_samples, self.num_samples))
            self.data.append(v_data)
        
    
    def PrepareDataLoader(self):
        c_data = np.zeros((len(self.variables), self.num_samples, 1, self.c[0], self.c[1], self.c[2]))
        c_label = np.zeros((len(self.labels), self.num_samples, 1, len(self.labels[0])))

        for i in range(self.num_samples):
            resample = True
            while(resample):
                resample = False
                x = np.random.randint(0, self.dim[0] - self.c[0])
                if self.args.dataset == 'ionization':
                    x = np.random.randint(260, self.dim[0] - self.c[0])
                if self.args.dataset == 'combustion':
                    x = np.random.randint(90, self.dim[0] - self.c[0] - 90)
                y = np.random.randint(0, self.dim[1] - self.c[1])
                z = np.random.randint(0, self.dim[2] - self.c[2])
                
                for j in range(len(self.variables)):
                    c_data[j][i][0] = self.data[j][i][x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
                    if np.std(c_data[j][i][0]) == 0.0:
                        resample = True
                        break
                    c_label[j][i][0] = self.labels[j]
        
        c_data = torch.FloatTensor(c_data)
        c_label = torch.FloatTensor(c_label)
        
        if self.pretrain:
            c_label = torch.cat([c_label[x] for x in range(len(self.variables))])
            c_source = torch.cat([c_data[x] for x in range(len(self.variables))])
            c_target = torch.cat([c_data[x] for x in range(len(self.variables))])
        else:
            c_label = torch.cat([c_label[x] for x in range(1,len(self.variables))])
            c_source = torch.cat([c_data[0]]*(len(self.variables)-1))
            c_target = torch.cat([c_data[x] for x in range(1,len(self.variables))])

        dataset = torch.utils.data.TensorDataset(c_source, c_target, c_label)

        kwargs = {'num_workers': 1, 'pin_memory': True}
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, **kwargs)
        return data_loader

def fod(tensor,dim):
	B,H,L,W = tensor.size()[0], tensor.size()[-3], tensor.size()[-2], tensor.size()[-1]
	tensor = torch.squeeze(tensor, 1)
	C = 1
	diff = torch.full([B,C,L,H,W],2,dtype=torch.int).cuda()
	if dim == 1:
		tensor_h = torch.cat((tensor[:,1:H,:,:,],tensor[:,H-1:H,:,:,]),dim=1)
		tensor_h_ = torch.cat((tensor[:,0:1,:,:,],tensor[:,0:H-1,:,:,]),dim=1)
		diff[:,0:1,:,:,] = 1
		diff[:,H-1:H,:,:,] = 1
	elif dim == 2:
		tensor_h = torch.cat((tensor[:,:,1:L,:,],tensor[:,:,L-1:L,:,]),dim=2)
		tensor_h_ = torch.cat((tensor[:,:,0:1,:,],tensor[:,:,0:L-1,:,]),dim=2)
		diff[:,:,0:1,:,] = 1
		diff[:,:,L-1:L,:,] = 1
	elif dim == 3:
		tensor_h = torch.cat((tensor[:,:,:,1:W],tensor[:,:,:,W-1:W]),dim=3)
		tensor_h_ = torch.cat((tensor[:,:,:,0:1],tensor[:,:,:,0:W-1]),dim=3)
		diff[:,:,:,0:1] = 1
		diff[:,:,:,W-1:W] = 1
	else:
		assert "Not implemented!"

	derivative = tensor_h - tensor_h_

	return derivative/diff

def gaussian3d(device=torch.device('cpu'), dtype=torch.float32):
    gauss2d = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1.]], device=device).to(dtype) / 16.0
    return torch.stack([gauss2d, 2*gauss2d, gauss2d]) / 4.0

def get_gaussian1d(size, sigma, dtype=torch.float32):
    coords = torch.arange(size)
    coords-= size//2

    gauss = torch.exp(- coords**2 / (2*sigma**2))
    gauss/= gauss.sum()
    return gauss.to(dtype)

def filter_gaussian_separated(input, win):
    win = win.to(input.dtype).to(input.device)
    out = F.conv3d(input, win,                groups=input.size(1))
    out = F.conv3d(out,   win.transpose(3,4), groups=input.size(1))
    out = F.conv3d(out,   win.transpose(2,4), groups=input.size(1))
    return out

def ssim3d(pred, targ, data_range=1.0, win_size=11, sigma=1.5, non_negative=True, return_average=True):
    N, C, W, H, D = pred.shape
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range)**2, (K2 * data_range)**2

    # win = gaussian3d(device=pred.device, dtype=pred.dtype)[None,None]
    win = get_gaussian1d(win_size, sigma, dtype=pred.dtype).to(pred.device)[None, None, None, None]
    mu1, mu2 = filter_gaussian_separated(pred, win), filter_gaussian_separated(targ, win)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_gaussian_separated(pred * pred, win) - mu1_sq
    sigma2_sq = filter_gaussian_separated(targ * targ, win) - mu2_sq
    sigma12   = filter_gaussian_separated(pred * targ, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    if non_negative: cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if return_average: return ssim_map.mean()
    else:              return ssim_map

def ssim1d(pred, targ, data_range=1.0, win_size=11, sigma=1.5, non_negative=True, return_average=True):
    N, C, W, H, D = pred.shape
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range)**2, (K2 * data_range)**2

    # win = gaussian3d(device=pred.device, dtype=pred.dtype)[None,None]
    win = get_gaussian1d(win_size, sigma, dtype=pred.dtype).to(pred.device)[None, None]
    mu1, mu2 = F.conv1d(pred, win, groups=pred.size(1)), F.conv1d(targ, win, groups=pred.size(1))

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv1d(pred * pred, win, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv1d(targ * targ, win, groups=targ.size(1)) - mu2_sq
    sigma12   = F.conv1d(pred * targ, win, groups=pred.size(1)) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    if non_negative: cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if return_average: return ssim_map.mean()
    else:              return ssim_map
