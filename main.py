####### This is the code for DeepPTZ (WACV2019) #######
import argparse
import pdb
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torch.optim as optim
from torch.autograd import Variable
import torch.optim as optim
import random
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import math
import shutil
import datetime
import csv
import torchvision
from torchvision import datasets, models, transforms
from myinception import MyInception3_siamese
from myinception_efficient import MyInception3_siamese_efficient
from transforms3d.euler import EulerFuncs
from utils import time_string


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataDIR', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-g', '--gpu', dest='gpu', default=None,
                    help='gpu allocate')
parser.add_argument('--cp', '--checkpoint', dest='checkpoint', default=None,
                    help='path to pre-trained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--comment', dest='comment', default=None,
                    help='evaluate model on validation set')
parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                    help='debug mode')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='model pretrained parameters, different from checkpoint')
parser.add_argument('--pf', '--pretrained_fixed', dest='pretrained_fixed', action='store_true',
                    help='model pretrained parameters, fix it')
parser.add_argument('-s', '--siamese', dest='siamese', action='store_false',
                    help='siamese network')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lt', '--loss_type', default=1, type=int, metavar='N',
                    help='Decide loss type')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='random seed')
parser.add_argument('--fr', '--focal_ratio', default=10, type=int, metavar='N',
                    help='Decide loss type')
parser.add_argument('--milestones', default=[5,10,15,20,25], metavar='N', nargs='*', 
                    help='epochs at which learning rate is divided by 2')
parser.add_argument('--ls', '--lr_decay_step', default=5, type=int, metavar='N',
                    help='lr is divided by 2 after ls epochs')
parser.add_argument('--efficient', action='store_true',
                    help='To run the model in an efficient way for saving parameter and time')
args = parser.parse_args()
### Define which gpu to use, if you ahve multiple gpus ###
if parser.parse_args().gpu != None:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

class PTZCameraDataset(Dataset):
    """Face Targets dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Targets_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Targets_frame)

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.root_dir, self.Targets_frame.iloc[idx, 0])
        image1 = io.imread(img_name1)

        img_name2 = os.path.join(self.root_dir, self.Targets_frame.iloc[idx, 1])
        image2 = io.imread(img_name2)

        Targets = self.Targets_frame.iloc[idx, 2:12].values # 2,3,4,5,6,7,8,9,10,11

        Targets = Targets.astype('float') # .reshape(-1, 4)
        Targets[3] = (Targets[3]-mean_focal)/args.fr
        Targets[4] = (Targets[4]-mean_focal)/args.fr
        Targets[8] = (Targets[8]-0.5)*10
        Targets[9] = (Targets[9]-0.5)*10

        Targetsnew = np.zeros(14)
        Targetsnew[0:5] = Targets[0:5]
        Targetsnew[5] = Targets[8]
        Targetsnew[6] = Targets[9]
        Targetsnew[7:10] = Targets[5:8]
        Targetsnew[10] = Targets[4] 
        Targetsnew[11] = Targets[3] 
        Targetsnew[12] = Targets[9] 
        Targetsnew[13] = Targets[8] 
        #pdb.set_trace()
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample = {'image1': image1, 'image2': image2, 'Targets': Targetsnew}
        return sample

def show_image1(image1, image2, Targets):
    """Show image with Targets"""
    plt.imshow(image1)
    # plt.imshow(image2)
    # plt.scatter(Targets[:, 0], Targets[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))

def list_round(inputs):
    outputs = []
    for input in inputs:
        input_round = round(input, 3)
        outputs.append(input_round)
    return tuple(outputs)

def train(epoch, model, train_loader, optimizer):
    args = parser.parse_args()
    args.debug = 1
    model.train()
    losses = AverageMeter()
    roll1_losses  = AverageMeter()
    pitch1_losses = AverageMeter()
    yaw1_losses   = AverageMeter()
    focal11_losses = AverageMeter()
    focal12_losses = AverageMeter()
    dist11_losses = AverageMeter()
    dist12_losses = AverageMeter()
    roll2_losses  = AverageMeter()
    pitch2_losses = AverageMeter()
    yaw2_losses   = AverageMeter()
    focal21_losses = AverageMeter()
    focal22_losses = AverageMeter()
    dist21_losses = AverageMeter()
    dist22_losses = AverageMeter()
    focal1avg_losses = AverageMeter()
    focal2avg_losses = AverageMeter()
    dist1avg_losses = AverageMeter()
    dist2avg_losses = AverageMeter()
    #pdb.set_trace()
    for batch_idx, sample in enumerate(train_loader):
        model.train()
        Image1 = sample['image1']
        Image1 = Image1.float().to(device)
        Image2 = sample['image2']
        Image2 = Image2.float().to(device)

        target = sample['Targets']
        target = target.float().to(device)   # double to float

        output = model(Image1, Image2)


        roll1_label_cont, pitch1_label_cont, yaw1_label_cont, focal1_label_cont, focal2_label_cont, dist1_label_cont, dist2_label_cont, roll2_label_cont, pitch2_label_cont, yaw2_label_cont = target[:,0], target[:,1], target[:,2], target[:,3]*args.fr+mean_focal, target[:,4]*args.fr+mean_focal, target[:,5]/10+0.5, target[:,6]/10+0.5, target[:,7], target[:,8], target[:,9] #torch.chunk(target, 3, dim=1)   

        roll1_predicted  = output[:,0]
        pitch1_predicted = output[:,1]
        yaw1_predicted   = output[:,2]
        focal11_predicted = output[:,3]*args.fr+mean_focal
        focal12_predicted = output[:,4]*args.fr+mean_focal
        dist11_predicted = output[:,5]/10 + 0.5
        dist12_predicted = output[:,6]/10 + 0.5
        roll2_predicted  = output[:,7]
        pitch2_predicted = output[:,8]
        yaw2_predicted   = output[:,9]
        focal22_predicted = output[:,10]*args.fr+mean_focal
        focal21_predicted = output[:,11]*args.fr+mean_focal
        dist22_predicted = output[:,12]/10 + 0.5
        dist21_predicted = output[:,13]/10 + 0.5
        focal1avg_predicted = (focal11_predicted + focal21_predicted)/2
        focal2avg_predicted = (focal12_predicted + focal22_predicted)/2
        dist1avg_predicted = (dist11_predicted + dist21_predicted)/2
        dist2avg_predicted = (dist12_predicted + dist22_predicted)/2
        loss = loss_type(output, target)


        roll1_loss = l1loss(roll1_predicted, roll1_label_cont)
        roll1_losses.update(roll1_loss.item(), roll1_predicted.size(0))
        pitch1_loss = l1loss(pitch1_predicted, pitch1_label_cont)
        pitch1_losses.update(pitch1_loss.item(), pitch1_predicted.size(0))        
        yaw1_loss = l1loss(yaw1_predicted, yaw1_label_cont)
        yaw1_losses.update(yaw1_loss.item(), yaw1_predicted.size(0))
        avg1_losses = (roll1_losses.avg+pitch1_losses.avg+yaw1_losses.avg)/3
        focal11_loss = l1loss(focal11_predicted, focal1_label_cont)
        focal11_losses.update(focal11_loss.item(), focal11_predicted.size(0))
        focal12_loss = l1loss(focal12_predicted, focal2_label_cont)
        focal12_losses.update(focal12_loss.item(), focal12_predicted.size(0))
        dist11_loss = l1loss(dist11_predicted, dist1_label_cont)
        dist11_losses.update(dist11_loss.item(), dist11_predicted.size(0))
        dist12_loss = l1loss(dist12_predicted, dist2_label_cont)
        dist12_losses.update(dist12_loss.item(), dist12_predicted.size(0))
        roll2_loss = l1loss(roll2_predicted, roll2_label_cont)
        roll2_losses.update(roll2_loss.item(), roll2_predicted.size(0))
        pitch2_loss = l1loss(pitch2_predicted, pitch2_label_cont)
        pitch2_losses.update(pitch2_loss.item(), pitch2_predicted.size(0))        
        yaw2_loss = l1loss(yaw2_predicted, yaw2_label_cont)
        yaw2_losses.update(yaw2_loss.item(), yaw2_predicted.size(0))
        avg2_losses = (roll2_losses.avg+pitch2_losses.avg+yaw2_losses.avg)/3
        focal21_loss = l1loss(focal21_predicted, focal1_label_cont)
        focal21_losses.update(focal21_loss.item(), focal21_predicted.size(0))
        focal22_loss = l1loss(focal22_predicted, focal2_label_cont)
        focal22_losses.update(focal22_loss.item(), focal22_predicted.size(0))
        dist21_loss = l1loss(dist21_predicted, dist1_label_cont)
        dist21_losses.update(dist21_loss.item(), dist21_predicted.size(0))
        dist22_loss = l1loss(dist22_predicted, dist2_label_cont)
        dist22_losses.update(dist22_loss.item(), dist22_predicted.size(0))
        focal1avg_loss = l1loss(focal1avg_predicted, focal1_label_cont)
        focal1avg_losses.update(focal1avg_loss.item(), focal1avg_predicted.size(0))
        focal2avg_loss = l1loss(focal2avg_predicted, focal2_label_cont)
        focal2avg_losses.update(focal2avg_loss.item(), focal2avg_predicted.size(0))
        dist1avg_loss = l1loss(dist1avg_predicted, dist1_label_cont)
        dist1avg_losses.update(dist1avg_loss.item(), dist1avg_predicted.size(0))
        dist2avg_loss = l1loss(dist2avg_predicted, dist2_label_cont)
        dist2avg_losses.update(dist2avg_loss.item(), dist2avg_predicted.size(0))
        losses.update(loss.item(), target.size(0))
        if batch_idx % 10 == 0:
            print('{:s}Train Epoch: {} [{}/{}] \troll1: {:.3f}\tpitch1: {:.3f}\tyaw1: {:.3f}\tavg1: {:.3f}\tfocal11: {:.3f}\tfocal12: {:.3f}\tdist11: {:.3f}\tdist12: {:.3f}\troll2: {:.3f}\tpitch2: {:.3f}\tyaw2: {:.3f}\tavg2: {:.3f}\tfocal21: {:.3f}\tfocal22: {:.3f}\tdist21: {:.3f}\tdist22: {:.3f}\tfocal1avg: {:.3f}\tfocal2avg: {:.3f}\tdist1avg: {:.3f}\tdist2avg: {:.3f}\tloss: {:.3f}'.format(
                time_string(), epoch, batch_idx*batch_size, len(train_loader.dataset), roll1_losses.avg, pitch1_losses.avg, yaw1_losses.avg, avg1_losses, focal11_losses.avg, focal12_losses.avg, dist11_losses.avg, dist12_losses.avg, roll2_losses.avg, pitch2_losses.avg, yaw2_losses.avg, avg2_losses, focal21_losses.avg, focal22_losses.avg, dist21_losses.avg, dist22_losses.avg, focal1avg_losses.avg, focal2avg_losses.avg, dist1avg_losses.avg, dist2avg_losses.avg, losses.avg))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(args.debug, batch_idx)
        if args.debug:
            if batch_idx == 50: break
    return roll1_losses.avg, pitch1_losses.avg, yaw1_losses.avg, avg1_losses, focal11_losses.avg, focal12_losses.avg, dist11_losses.avg, dist12_losses.avg, roll2_losses.avg, pitch2_losses.avg, yaw2_losses.avg, avg2_losses, focal21_losses.avg, focal22_losses.avg, dist21_losses.avg, dist22_losses.avg, focal1avg_losses.avg, focal2avg_losses.avg, dist1avg_losses.avg, dist2avg_losses.avg, losses.avg

def val(epoch, model, val_loader):
    args = parser.parse_args()
    model.eval() ## under the eval mode, there is only one output whitout the aux output
    #val_losses = AverageMeter()
    roll1_losses  = AverageMeter()
    pitch1_losses = AverageMeter()
    yaw1_losses   = AverageMeter()
    focal11_losses = AverageMeter()
    focal12_losses = AverageMeter()
    dist11_losses = AverageMeter()
    dist12_losses = AverageMeter()
    roll2_losses  = AverageMeter()
    pitch2_losses = AverageMeter()
    yaw2_losses   = AverageMeter()
    focal21_losses = AverageMeter()
    focal22_losses = AverageMeter()
    dist21_losses = AverageMeter()
    dist22_losses = AverageMeter()
    focal1avg_losses = AverageMeter()
    focal2avg_losses = AverageMeter()
    dist1avg_losses = AverageMeter()
    dist2avg_losses = AverageMeter()
    for val_idx, sample in enumerate(val_loader):
        Image1 = sample['image1']
        Image1 = Image1.float().to(device)
        Image2 = sample['image2']
        Image2 = Image2.float().to(device)
        target = sample['Targets']
        target = target.float().to(device)
        
        output = model(Image1, Image2)
        
        roll1_label_cont, pitch1_label_cont, yaw1_label_cont, focal1_label_cont, focal2_label_cont, dist1_label_cont, dist2_label_cont, roll2_label_cont, pitch2_label_cont, yaw2_label_cont = target[:,0], target[:,1], target[:,2], target[:,3]*args.fr+mean_focal, target[:,4]*args.fr+mean_focal, target[:,5]/10+0.5, target[:,6]/10+0.5, target[:,7], target[:,8], target[:,9] #torch.chunk(target, 3, dim=1)  

        roll1_predicted  = output[:,0]
        pitch1_predicted = output[:,1]
        yaw1_predicted   = output[:,2]
        focal11_predicted = output[:,3]*args.fr+mean_focal
        focal12_predicted = output[:,4]*args.fr+mean_focal
        dist11_predicted = output[:,5]/10 + 0.5
        dist12_predicted = output[:,6]/10 + 0.5
        roll2_predicted  = output[:,7]
        pitch2_predicted = output[:,8]
        yaw2_predicted   = output[:,9]
        focal22_predicted = output[:,10]*args.fr+mean_focal
        focal21_predicted = output[:,11]*args.fr+mean_focal
        dist22_predicted = output[:,12]/10 + 0.5
        dist21_predicted = output[:,13]/10 + 0.5
        focal1avg_predicted = (focal11_predicted + focal21_predicted)/2
        focal2avg_predicted = (focal12_predicted + focal22_predicted)/2
        dist1avg_predicted = (dist11_predicted + dist21_predicted)/2
        dist2avg_predicted = (dist12_predicted + dist22_predicted)/2

        roll1_loss = l1loss(roll1_predicted, roll1_label_cont)
        roll1_losses.update(roll1_loss.item(), roll1_predicted.size(0))
        pitch1_loss = l1loss(pitch1_predicted, pitch1_label_cont)
        pitch1_losses.update(pitch1_loss.item(), pitch1_predicted.size(0))        
        yaw1_loss = l1loss(yaw1_predicted, yaw1_label_cont)
        yaw1_losses.update(yaw1_loss.item(), yaw1_predicted.size(0))
        avg1_losses = (roll1_losses.avg+pitch1_losses.avg+yaw1_losses.avg)/3
        focal11_loss = l1loss(focal11_predicted, focal1_label_cont)
        focal11_losses.update(focal11_loss.item(), focal11_predicted.size(0))
        focal12_loss = l1loss(focal12_predicted, focal2_label_cont)
        focal12_losses.update(focal12_loss.item(), focal12_predicted.size(0))
        dist11_loss = l1loss(dist11_predicted, dist1_label_cont)
        dist11_losses.update(dist11_loss.item(), dist11_predicted.size(0))
        dist12_loss = l1loss(dist12_predicted, dist2_label_cont)
        dist12_losses.update(dist12_loss.item(), dist12_predicted.size(0))
        roll2_loss = l1loss(roll2_predicted, roll2_label_cont)
        roll2_losses.update(roll2_loss.item(), roll2_predicted.size(0))
        pitch2_loss = l1loss(pitch2_predicted, pitch2_label_cont)
        pitch2_losses.update(pitch2_loss.item(), pitch2_predicted.size(0))        
        yaw2_loss = l1loss(yaw2_predicted, yaw2_label_cont)
        yaw2_losses.update(yaw2_loss.item(), yaw2_predicted.size(0))
        avg2_losses = (roll2_losses.avg+pitch2_losses.avg+yaw2_losses.avg)/3
        focal21_loss = l1loss(focal21_predicted, focal1_label_cont)
        focal21_losses.update(focal21_loss.item(), focal21_predicted.size(0))
        focal22_loss = l1loss(focal22_predicted, focal2_label_cont)
        focal22_losses.update(focal22_loss.item(), focal22_predicted.size(0))
        dist21_loss = l1loss(dist21_predicted, dist1_label_cont)
        dist21_losses.update(dist21_loss.item(), dist21_predicted.size(0))
        dist22_loss = l1loss(dist22_predicted, dist2_label_cont)
        dist22_losses.update(dist22_loss.item(), dist22_predicted.size(0))
        focal1avg_loss = l1loss(focal1avg_predicted, focal1_label_cont)
        focal1avg_losses.update(focal1avg_loss.item(), focal1avg_predicted.size(0))
        focal2avg_loss = l1loss(focal2avg_predicted, focal2_label_cont)
        focal2avg_losses.update(focal2avg_loss.item(), focal2avg_predicted.size(0))
        dist1avg_loss = l1loss(dist1avg_predicted, dist1_label_cont)
        dist1avg_losses.update(dist1avg_loss.item(), dist1avg_predicted.size(0))
        dist2avg_loss = l1loss(dist2avg_predicted, dist2_label_cont)
        dist2avg_losses.update(dist2avg_loss.item(), dist2avg_predicted.size(0))
        if val_idx % 10 == 0:
            print('Valid Epoch: {} [{}/{}] \troll1: {:.3f}\tpitch1: {:.3f}\tyaw1: {:.3f}\tavg1: {:.3f}\tfocal11: {:.3f}\tfocal12: {:.3f}\tdist11: {:.3f}\tdist12: {:.3f}\troll2: {:.3f}\tpitch2: {:.3f}\tyaw2: {:.3f}\tavg2: {:.3f}\tfocal21: {:.3f}\tfocal22: {:.3f}\tdist21: {:.3f}\tdist22: {:.3f}\tfocal1avg: {:.3f}\tfocal2avg: {:.3f}\tdist1avg: {:.3f}\tdist2avg: {:.3f}'.format(epoch, val_idx*batch_size, len(val_loader.dataset), roll1_losses.avg, pitch1_losses.avg, yaw1_losses.avg, avg1_losses, focal11_losses.avg, focal12_losses.avg, dist11_losses.avg, dist12_losses.avg, roll2_losses.avg, pitch2_losses.avg, yaw2_losses.avg, avg2_losses, focal21_losses.avg, focal22_losses.avg, dist21_losses.avg, dist22_losses.avg, focal1avg_losses.avg, focal2avg_losses.avg, dist1avg_losses.avg, dist2avg_losses.avg))
        if args.debug:
            if val_idx == 50: break
    print(' ******** Validation Loss: *\troll1: {:.3f}\tpitch1: {:.3f}\tyaw1: {:.3f}\tavg1: {:.3f}\tfocal11: {:.3f}\tfocal12: {:.3f}\tdist11: {:.3f}\tdist12: {:.3f}\troll2: {:.3f}\tpitch2: {:.3f}\tyaw2: {:.3f}\tavg2: {:.3f}\tfocal21: {:.3f}\tfocal22: {:.3f}\tdist21: {:.3f}\tdist22: {:.3f}\tfocal1avg: {:.3f}\tfocal2avg: {:.3f}\tdist1avg: {:.3f}\tdist2avg: {:.3f}********'.format(roll1_losses.avg, pitch1_losses.avg, yaw1_losses.avg, avg1_losses, focal11_losses.avg, focal12_losses.avg, dist11_losses.avg, dist12_losses.avg, roll2_losses.avg, pitch2_losses.avg, yaw2_losses.avg, avg2_losses, focal21_losses.avg, focal22_losses.avg, dist21_losses.avg, dist22_losses.avg, focal1avg_losses.avg, focal2avg_losses.avg, dist1avg_losses.avg, dist2avg_losses.avg))
    return roll1_losses.avg, pitch1_losses.avg, yaw1_losses.avg, avg1_losses, focal11_losses.avg, focal12_losses.avg, dist11_losses.avg, dist12_losses.avg, roll2_losses.avg, pitch2_losses.avg, yaw2_losses.avg, avg2_losses, focal21_losses.avg, focal22_losses.avg, dist21_losses.avg, dist22_losses.avg, focal1avg_losses.avg, focal2avg_losses.avg, dist1avg_losses.avg, dist2avg_losses.avg

# Training settings
if args.efficient:
    batch_size = 32 # Ween GPU1080Ti is used, we used 32 by defualt 
else:
    batch_size = 16 # Ween GPU1080Ti is used, bs=32 will cause memorry issue
num_workers = 8

best_avg = math.inf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss().cuda(device)
reg_criterion = torch.nn.MSELoss().cuda(device)
softmax = torch.nn.Softmax(dim=1).cuda(device)
maxium_degree = 15
mean_focal = 275 #150
maxium_focal = 225 #100
maxium_focal = maxium_focal/args.fr
l1loss = torch.nn.L1Loss(reduce = True).cuda(device)
args = parser.parse_args()

#### We tried with different loss function, SmoothL1loss lead to the best performance, thus it was used in the paper  #####
if args.lt == 1:
    loss_type = torch.nn.SmoothL1Loss(reduce=True).cuda(device)
elif args.lt == 2:
    loss_type = torch.nn.L1Loss(reduce=True).cuda(device)
elif args.lt ==3:
    loss_type = torch.nn.MSELoss(reduce=True).cuda(device)

alpha = 1
def main():    
    global best_avg, save_path
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    if args.debug:
        save_path = os.path.join('work', 'debug')
    else:
        save_path = os.path.join('work', timestamp+args.comment)

    if not os.path.exists(save_path): os.makedirs(save_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize, ])

    data_dir = args.dataDIR

    Train_dataset = PTZCameraDataset(csv_file=data_dir+'All_images/focal225degree15/Train/Train.csv',
                                        root_dir=data_dir, transform=transform)

    Val_dataset = PTZCameraDataset(csv_file=data_dir + 'All_images/focal225degree15/Val/Val.csv',
                                        root_dir=data_dir, transform=transform)
    print('{} samples found, {} train samples and {} test samples '.format(len(Val_dataset)+len(Train_dataset),
                                                                           len(Train_dataset),
                                                                           len(Val_dataset)))

    train_loader = torch.utils.data.DataLoader(dataset=Train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=Val_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               shuffle=False)
    if args.efficient:
        model = MyInception3_siamese_efficient(siamese=args.siamese, pretrained = args.pretrained, pretrained_fixed=args.pretrained_fixed, aux_logits=False)
    else:
        model = MyInception3_siamese(siamese=args.siamese, pretrained = args.pretrained, pretrained_fixed=args.pretrained_fixed, aux_logits=False)

    if args.checkpoint:
       network_data = torch.load(args.checkpoint)
       model.load_state_dict(network_data['state_dict'])
    model = torch.nn.DataParallel(model).to(device)
    print('Total Parameters: %.3fM' %(sum(p.numel() for p in model.parameters())/1000000.0))

    cudnn.benchmark = True

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)

    if args.evaluate:
        with torch.no_grad():
            val_loss = val(0, model, val_loader)
        return


    for i in range(len(Train_dataset)):
        sample = Train_dataset[i]

        print(i, sample['image1'].shape, sample['image2'].shape, sample['Targets'].shape)

        # To check whether the image is properly processed 
        # ax = plt.subplot(1, 4, i + 1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        # show_image1(**sample)
        if i == 0:
            break
    with open(os.path.join(save_path,'progress_log_summary.csv'), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['t:train', 'v:val'])
        writer.writerow(['epoch', 'rollt1', 'pitcht1', 'yawt1', 'avgt1', 'focalt11', 'focalt12', 'distt11','distt12','rollt2', 'pitcht2', 'yawt2', 'avgt2', 'focalt21', 'focalt22', 'distt21','distt22', 'focalt1avg', 'focalt2avg', 'distt1avg', 'distt2avg', 'losst', 'rollv1', 'pitchv1', 'yawv1', 'avgv1', 'focalv11', 'focalv12', 'distv11','distv12', 'rollv2', 'pitchv2', 'yawv2', 'avgv2', 'focalv21', 'focalv22', 'distv21','distv22', 'focalv1avg', 'focalv2avg', 'distv1avg', 'distv2avg'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.ls*step for step in [1,2,3,4,5]], gamma=0.5)    
    for epoch in range(0, 6*args.ls):
        scheduler.step()
        roll1_train, pitch1_train, yaw1_train, avg1_train, focal11_train, focal12_train, dist11_train, dist12_train, roll2_train, pitch2_train, yaw2_train, avg2_train, focal21_train, focal22_train, dist21_train, dist22_train, focal1avg_train, focal2avg_train,dist1avg_train, dist2avg_train, train_loss = list_round(train(epoch, model, train_loader, optimizer))
        with torch.no_grad():
            roll1_val, pitch1_val, yaw1_val, avg1_val, focal11_val, focal12_val, dist11_val, dist12_val, roll2_val, pitch2_val, yaw2_val, avg2_val, focal21_val, focal22_val, dist21_val, dist22_val, focal1avg_val, focal2avg_val, dist1avg_val, dist2avg_val = list_round(val(epoch, model, val_loader))
        is_best = avg1_val < best_avg
        best_avg = min(best_avg, avg1_val)
        try:
            state_dict = model.module.state_dict() # model.DataParallel()
        except AttributeError:
            state_dict = model.state_dict()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'best_loss': best_avg,
        }, is_best)

        with open(os.path.join(save_path,'progress_log_summary.csv'), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epoch, roll1_train, pitch1_train, yaw1_train, avg1_train, focal11_train, focal12_train, dist11_train, dist12_train, roll2_train, pitch2_train, yaw2_train, avg2_train, focal21_train, focal22_train, dist21_train, dist22_train, focal1avg_train, focal2avg_train,dist1avg_train, dist2avg_train, train_loss, roll1_val, pitch1_val, yaw1_val, avg1_val, focal11_val, focal12_val, dist11_val, dist12_val, roll2_val, pitch2_val, yaw2_val, avg2_val, focal21_val, focal22_val, dist21_val, dist22_val, focal1avg_val, focal2avg_val, dist1avg_val, dist2avg_val])

if __name__ == '__main__':
    main()
