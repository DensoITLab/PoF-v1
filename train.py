#!/usr/bin/env python3

import torch, os, sys, math, argparse, time, datetime, pickle, random, itertools
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import copy
import re
from wide_resnet import Wide_ResNet_fet
from classifier import Classifier
from collections import OrderedDict
from PIL import Image
import multiprocessing
import math
from cutout import Cutout

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False


best_error = 100

def set_args():
    parser = argparse.ArgumentParser()
    # Network settings
    parser.add_argument('--network', type=str, default='wide_resnet', 
                        choices=('resnet18', 'wide_resnet'))
    parser.add_argument('--depth', type=int, default=28, help='depth of net') # WideResnet depth
    parser.add_argument('--factor', type=int, default=10, help='factor of net') # WideResnet width
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--std_clf', type=float, default=0.1)
    # Data settings
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=('cifar10', 'cifar100', 'svhn', 'fashion_mnist'))
    parser.add_argument('--flgDataAug', type=bool, default=1)
    parser.add_argument('--augment_type', type=str, default='basic', choices=('basic', 'cutout'),
                                          help='Basic (horizontal flip, padding by 4px, and random crop), Cutout (Devries & Taylor, 2017)') 
    parser.add_argument('--padding', type=float, default=4, help='padding size')
    parser.add_argument('--flgCutout', type=bool, default=0)
    parser.add_argument('--co_length', type=int, default=16, help='cutout size')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='momentumSGD', choices=('sgd', 'momentumSGD', 'adam'))
    parser.add_argument('--init_lr', type=float, default=0.1, help='learn rate')
    parser.add_argument('--lr_drop_rate', type=float, default=0.2, help='lr drop rate')
    parser.add_argument('--lr_step_schedule', type=int, default=[60, 120, 160, 200], help='schedule of learn rate') #global epoch
    parser.add_argument('--weight_decay_rate', type=float, default=5e-4, help='weight decay rate')
    parser.add_argument('--momentum_rate', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--flgNesterov', type=bool, default=1)
    # PoF settings
    parser.add_argument('--flgPoF', type=bool, default=0)
    parser.add_argument('--gamma', type=float, default=2.0) # gamma=2->taigan
    parser.add_argument('--weak_clf_batch_size', type=int, default=256) # For PoF argument
    parser.add_argument('--numPartition', type=int, default=16)
    parser.add_argument('--line_search_init_lr', type=float, default=1.0, help='learn rate of Line Search')
    parser.add_argument('--PoFSave', type=str, default='./result/save/pof') # rewrite!
    # Batch settings
    parser.add_argument('--train_total_batch_size', type=int, default=64)
    parser.add_argument('--test_total_batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size_per_gpu', type=int, default=64)
    parser.add_argument('--test_batch_size_per_gpu', type=int, default=64)
    # Settings for learning from  pretrained model
    parser.add_argument('--flgContinue', type=bool, default=0)
    parser.add_argument('--PretrainedSave', type=str, default='./result/save/pret') # rewrite!
    # GPU settings
    parser.add_argument('--world_size', default=1, type=int) # number of total GPU
    parser.add_argument('--idGPU', type=int, default=0) 
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--workers', default=4, type=int, metavar='N', 
                        help='number of data loading workers (default: 4)')
    # Seed settings
    parser.add_argument('--flgFixSeed', type=bool, default=0)
    parser.add_argument('--seed', type=int, default=0)
    # Epoch settings
    parser.add_argument('--start_epoch', type=int, default=200) # global epoch when continued
    parser.add_argument('--end_epoch', type=int, default=300) # global epoch
    # Other
    parser.add_argument('--flgDist', type=bool, default=0)
    parser.add_argument('--freqValid', type=int, default=5, help='epoch')
    parser.add_argument('--flgSaveModel', type=bool, default=1)
    parser.add_argument('--freqSave', type=int, default=10, help='epoch')
    parser.add_argument('--dirSave', type=str, default='./result/save/') # rewrite!
    
    args = parser.parse_args()
    
    return args

def print_config(args, train_loader, numParam):
    print("==============================================================")
    if args.flgContinue:
        print("Continue Train from {:d} epoch".format(args.start_epoch))
    else:
        print("New Train")
    print('Num Train Epoch: {}\tStep Lr schedule: {}'.format(args.end_epoch, args.lr_step_schedule))
    print('Learning rate: {}\tLearning Rate Drop Rate: {}'.format(args.init_lr, args.lr_drop_rate))
    print('Line Search Learning Rate: {}'.format(args.line_search_init_lr))
    print('Weight Decay Rate: {}\tClassifier Init Std: {}'.format(args.weight_decay_rate, args.std_clf))
    print('Minibatch-size: {}\t\tLine Search Minibatch-size: {}'.format(args.train_total_batch_size, args.weak_clf_batch_size*args.world_size))
    print('Network: {}\tNetwork Depth: {}\tFactor: {}'.format(args.network, args.depth, args.factor))
    print('Optimizer: {}\tNesterov flag: {}\tMomentum rate: {:.2f}'.format(args.optimizer, args.flgNesterov, args.momentum_rate))
    print('PoF flag: {}'.format(args.flgPoF))
    if args.flgPoF:
        print('Gamma: {}\tPartition Num: {}'.format(args.gamma, args.numPartition))
    print('Dataset: {}\tNum Train Data: {}'.format(args.dataset,len(train_loader.dataset)))
    print('Data Augmentation: {}'.format(args.augment_type))
    if args.augment_type=='cutout':
        print('Cutout Length: {}'.format(args.co_length))
    print('Params Number: {}'.format(numParam))
    print('Distribution Train: {}'.format(args.flgDist))
    print('==============================================================')
    print("train start",flush=True)
    f = open("{}config.txt".format(args.dirSave),"a")
    f.write("==============================================================\n")
    if args.flgContinue:
        f.write("Continue Train from {:d} epoch\n".format(args.start_epoch))
    else:
        f.write("New Train\n")
    f.write('Num Train Epoch: {}\tStep Lr schedule: {}\n'.format(args.end_epoch, args.lr_step_schedule))
    f.write('Learning rate: {}\tLearning Rate Drop Rate: {}\n'.format(args.init_lr, args.lr_drop_rate))
    f.write('Line Search Learning Rate: {}\n'.format(args.line_search_init_lr))
    f.write('Weight Decay Rate: {}\tClassifier Init Std: {}\n'.format(args.weight_decay_rate, args.std_clf))
    f.write('Minibatch-size: {}\t\tLine Search Minibatch-size: {}\n'.format(args.train_total_batch_size, args.weak_clf_batch_size*args.world_size))
    f.write('Network: {}\tNetwork Depth: {}\tFactor: {}\n'.format(args.network, args.depth, args.factor))
    f.write('Optimizer: {}\tNesterov flag: {}\tMomentum rate: {:.2f}\n'.format(args.optimizer, args.flgNesterov, args.momentum_rate))
    f.write('PoF flag: {}\n'.format(args.flgPoF))
    if args.flgPoF:
        f.write('Gamma: {}\tPartition Num: {}\n'.format(args.gamma, args.numPartition))
    f.write('Dataset: {}\tNum Train Data: {}\n'.format(args.dataset, len(train_loader.dataset)))
    f.write('Data Augmentation: {}'.format(args.augment_type))
    if args.augment_type=='cutout':
        f.write('Cutout Length: {}'.format(args.co_length))
    f.write('Params Number: {}\n'.format(numParam))
    f.write('Distribution Train: {}\tGPU id: {}\n'.format(args.flgDist, args.idGPU))
    f.write("==============================================================\n")
    f.close()

    # set save
    result = dict()
    result['train'] = open(os.path.join(args.dirSave, 'train.csv'), 'a')
    result['test'] = open(os.path.join(args.dirSave, 'test.csv'), 'a')
    return result

def set_network(args, numClass):
    net = dict()
    torch.cuda.set_device(args.idGPU)
    if args.network == 'wide_resnet':
        net['fet'] = Wide_ResNet_fet(args)
        net['clf'] = Classifier(net['fet'].numFeatureDim, numClass)
    net['fet'] = net['fet'].cuda(args.idGPU)
    net['clf'] = net['clf'].cuda(args.idGPU)

    optimizer = dict()
    if args.optimizer == 'momentumSGD':
        optimizer['fet'] = Optim.SGD(net['fet'].parameters(), lr=args.init_lr, momentum=args.momentum_rate, weight_decay=args.weight_decay_rate, nesterov=args.flgNesterov)
        optimizer['clf'] = Optim.SGD(net['clf'].parameters(), lr=args.init_lr, momentum=args.momentum_rate, weight_decay=args.weight_decay_rate, nesterov=args.flgNesterov)
    if args.flgDist:
        print('Multi Processing Distributed Data Parallel')
        process_group = dist.new_group([i for i in range(args.world_size)])
        net['fet'] = nn.SyncBatchNorm.convert_sync_batchnorm(net['fet'], process_group)
        net['clf'] = nn.SyncBatchNorm.convert_sync_batchnorm(net['clf'], process_group)
        net['fet'] = nn.parallel.DistributedDataParallel(net['fet'], device_ids=[args.idGPU], output_device=args.idGPU)
        net['clf'] = nn.parallel.DistributedDataParallel(net['clf'], device_ids=[args.idGPU], output_device=args.idGPU)

    if args.flgContinue:
        fet_state = torch.load("{}fet_{}.pth".format(args.PretrainedSave, args.start_epoch))
        if not args.flgDist:
            new_state_dict = OrderedDict()
            for k, v in fet_state['param'].items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            net['fet'].load_state_dict(new_state_dict)
        else:
            net['fet'].load_state_dict(fet_state['param'])
        optimizer['fet'].load_state_dict(fet_state['optim'])
        clf_state = torch.load("{}clf_{}.pth".format(args.PretrainedSave, args.start_epoch))
        if not args.flgDist:
            new_state_dict = OrderedDict()
            for k, v in clf_state['param'].items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            net['clf'].load_state_dict(new_state_dict)
        else:
            net['clf'].load_state_dict(clf_state['param'])
        optimizer['clf'].load_state_dict(clf_state['optim'])
        optimizer['clf'].param_groups[0]['lr'] = args.init_lr
        if args.rank == 0:
            if not args.flgPoF:
                # Continue Standard Training
                args.dirSave = "./result/e2e/result_{}_{}_{}_".format(args.network, args.optimizer,args.dataset) + "{0:%Y%m%d_%H%M%S}/".format(datetime.datetime.now())
            else:
                # PoF Training
                args.dirSave = "./result/post/result_{}_PoF_{}_".format(args.network, args.dataset) + "{0:%Y%m%d_%H%M%S}/".format(datetime.datetime.now())
            os.mkdir(args.dirSave)
    else:
        if args.rank == 0:
            args.dirSave = "./result/e2e/result_{}_{}_{}_".format(args.network, args.optimizer,args.dataset) + "{0:%Y%m%d_%H%M%S}/".format(datetime.datetime.now())
            os.mkdir(args.dirSave)
        args.start_epoch = 0

    if args.flgPoF:
        if args.flgDist:
            net['weight'] = net['clf'].module.classifiers[-1].weight.data.clone().cuda()
            net['bias'] = net['clf'].module.classifiers[-1].bias.data.clone().cuda()
        else:
            net['weight'] = net['clf'].classifiers[-1].weight.data.clone().cuda()
            net['bias'] = net['clf'].classifiers[-1].bias.data.clone().cuda()

    net['numParam'] = sum([p.data.nelement() for p in net['fet'].parameters()]) + sum([p.data.nelement() for p in net['clf'].parameters()])

    return net, optimizer

def adjust_learning_rate(args, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = args.init_lr
    for milestone in args.lr_step_schedule:
        lr *= args.lr_drop_rate if epoch >= milestone else 1.0
    optimizer['fet'].param_groups[0]['lr'] = lr
    if not args.flgPoF:
        optimizer['clf'].param_groups[0]['lr'] = lr 


def test(test_loader, net, criterion, args):
    net['fet'].eval()
    net['clf'].eval()
    test_loss = 0
    test_error = 0
    total = 0
    iteration = 0
    with torch.no_grad():
        for idxBatch, (input, label) in enumerate(test_loader):
            input, label = input.cuda(args.idGPU, non_blocking=True), label.cuda(args.idGPU, non_blocking=True)
            feature = net['fet'](input)
            output = net['clf'](feature)
            test_loss += criterion(output, label).data.item()
            pred = output.data.max(1)[1]
            test_error += pred.ne(label.data).cpu().sum()
            total += label.size(0)
            iteration += 1

    # Average
    if args.flgDist:
        test_loss = torch.Tensor([test_loss]).cuda(args.idGPU)
        test_error = torch.Tensor([test_error]).cuda(args.idGPU)
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_error, op=dist.ReduceOp.SUM)
        test_loss = test_loss.item()/ args.world_size
        test_error = test_error.item() / args.world_size

    test_loss /= iteration
    test_error = 100.*float(test_error) / total

    return test_loss, test_error

def batch_for_weak_clf(dataset, batch_size):
    input_list = []
    label_list = []
    if args.flgDist:
        train_num = int(len(dataset) / args.world_size)
        idxData_list = np.random.randint(args.rank*train_num, (args.rank+1)*train_num, batch_size)
    else:
        idxData_list = np.random.randint(0, len(dataset), batch_size)
    for idxData in idxData_list:
        input, label = dataset[idxData]
        input_list.append(input)
        label_list.append(label)
    input = torch.stack(input_list)
    label = torch.tensor(label_list)

    return input, label

def generate_weak_clf(args, input, label, net, criterion, optimizer):
    # for DDP
    if args.flgDist:
        net['clf'].module.classifiers[-1].weight.data = net['weight'].clone()
        net['clf'].module.classifiers[-1].bias.data = net['bias'].clone()

        for m in optimizer['clf'].state.values():
                m['momentum_buffer'].zero_()
        
        # Calc mini-batch gradient
        input, label = input.cuda(args.idGPU, non_blocking=True), label.cuda(args.idGPU, non_blocking=True)
        with torch.no_grad():
            feature = net['fet'](input)
        optimizer['clf'].zero_grad()
        output = net['clf'](feature)
        loss = criterion(output, label)
        loss.backward()

        # Save Loss
        loss_org = loss.data.item()
        lr_search = args.line_search_init_lr

        # Linear search on mini-batch loss
        while True:
            net['clf'].module.classifiers[-1].weight.data -= lr_search * net['clf'].module.classifiers[-1].weight.grad
            net['clf'].module.classifiers[-1].bias.data -= lr_search * net['clf'].module.classifiers[-1].bias.grad
            output = net['clf'](feature)
            loss = criterion(output, label)
            net['clf'].module.classifiers[-1].weight.data = net['weight'].clone()
            net['clf'].module.classifiers[-1].bias.data = net['bias'].clone()
            if loss.data.item() > loss_org:
                break
            else:
                lr_search *= 2                
            if lr_search >= 10000:
                # Reject
                if args.bre:
                    break
                else:
                    clf_lr = 0
                    return clf_lr
        lr_line = torch.linspace(0, lr_search, args.numPartition+1)[1:]
        loss_buffer = torch.zeros(args.numPartition)
        for i in range(len(lr_line)):
            net['clf'].module.classifiers[-1].weight.data -= lr_line[i] * net['clf'].module.classifiers[-1].weight.grad
            net['clf'].module.classifiers[-1].bias.data -= lr_line[i] * net['clf'].module.classifiers[-1].bias.grad
            output = net['clf'](feature)
            loss = criterion(output, label)
            loss_buffer[i] = loss.data.item()
            net['clf'].module.classifiers[-1].weight.data = net['weight'].clone()
            net['clf'].module.classifiers[-1].bias.data = net['bias'].clone()
        min_loss_idx = torch.argmin(loss_buffer)
        uni = torch.rand(1)
        args.gamma = uni.item() * args.sampling_range
        clf_lr = args.gamma * lr_line[min_loss_idx]
        clf_grad = dict()
        clf_grad['weight'] = net['clf'].module.classifiers[-1].weight.grad.clone()
        clf_grad['bias'] = net['clf'].module.classifiers[-1].bias.grad.clone()

    # for Single GPU
    else:
        net['clf'].classifiers[-1].weight.data = net['weight'].clone()
        net['clf'].classifiers[-1].bias.data = net['bias'].clone()

        for m in optimizer['clf'].state.values():
                m['momentum_buffer'].zero_()

        # Calc mini-batch gradient
        input, label = input.cuda(args.idGPU, non_blocking=True), label.cuda(args.idGPU, non_blocking=True)
        with torch.no_grad():
            feature = net['fet'](input)
        optimizer['clf'].zero_grad()
        output = net['clf'](feature)
        loss = criterion(output, label)
        loss.backward()

        loss_org = loss.data.item()
        lr_search = args.line_search_init_lr
        
        # Linear search on mini-batch loss
        while True:
            net['clf'].classifiers[-1].weight.data -= lr_search * net['clf'].classifiers[-1].weight.grad
            net['clf'].classifiers[-1].bias.data -= lr_search * net['clf'].classifiers[-1].bias.grad
            output = net['clf'](feature)
            loss = criterion(output, label)
            net['clf'].classifiers[-1].weight.data = net['weight'].clone()
            net['clf'].classifiers[-1].bias.data = net['bias'].clone()
            if loss.data.item() > loss_org:
                break
            else:
                lr_search *= 2

            if lr_search >= 100:
                break
        lr_line = torch.linspace(0, lr_search, args.numPartition+1)[1:]
        loss_buffer = torch.zeros(args.numPartition)
        for i in range(len(lr_line)):
            net['clf'].classifiers[-1].weight.data -= lr_line[i] * net['clf'].classifiers[-1].weight.grad
            net['clf'].classifiers[-1].bias.data -= lr_line[i] * net['clf'].classifiers[-1].bias.grad
            output = net['clf'](feature)
            loss = criterion(output, label)
            loss_buffer[i] = loss.data.item()
            net['clf'].classifiers[-1].weight.data = net['weight'].clone()
            net['clf'].classifiers[-1].bias.data = net['bias'].clone()
        min_loss_idx = torch.argmin(loss_buffer)
        uni = torch.rand(1)
        args.gamma = uni.item() * args.sampling_range
        clf_lr = args.gamma * lr_line[min_loss_idx]
    
        clf_grad = dict()
        clf_grad['weight'] = net['clf'].classifiers[-1].weight.grad.clone()
        clf_grad['bias'] = net['clf'].classifiers[-1].bias.grad.clone()

    return clf_lr


def ConstantLoss(outputs, targets, idGPU):

    return torch.norm(outputs-outputs)


def train(train_loader, train_loader_forPoF, net, criterion, optimizer, args):
    net['fet'].train()
    net['clf'].train()
    train_loss = 0
    train_error = 0
    total = 0
    iteration = 0
    reject_counter = 0
    perturbation_norm = 0
    clf_lr = 1
    for idxBatch, (input, label) in enumerate(train_loader):
        input, label = input.cuda(args.idGPU, non_blocking=True), label.cuda(args.idGPU, non_blocking=True)
        
        if args.flgPoF:
            input_weak, label_weak = batch_for_weak_clf(train_loader_forPoF.dataset, args.weak_clf_batch_size)
            clf_lr = generate_weak_clf(args, input_weak, label_weak, net, criterion, optimizer) 
            if clf_lr == 0:
                # Reject
                reject_counter +=1
            if args.flgDist:
                perturbation_norm += torch.cat([clf_lr * net['clf'].module.classifiers[-1].bias.grad.unsqueeze(dim=1), clf_lr * net['clf'].module.classifiers[-1].weight.grad], dim=1).norm()
                net['clf'].module.classifiers[-1].weight.data -= clf_lr * net['clf'].module.classifiers[-1].weight.grad
                net['clf'].module.classifiers[-1].bias.data -= clf_lr * net['clf'].module.classifiers[-1].bias.grad
            else:
                perturbation_norm += torch.cat([clf_lr * net['clf'].classifiers[-1].bias.grad.unsqueeze(dim=1), clf_lr * net['clf'].classifiers[-1].weight.grad], dim=1).norm()
                net['clf'].classifiers[-1].weight.data -= clf_lr * net['clf'].classifiers[-1].weight.grad
                net['clf'].classifiers[-1].bias.data -= clf_lr * net['clf'].classifiers[-1].bias.grad

        feature = net['fet'](input)
        output = net['clf'](feature)
        loss = criterion(output, label)
        optimizer['fet'].zero_grad()
        optimizer['clf'].zero_grad()
        loss.backward()
        optimizer['clf'].step()
            
        # loss and error rate            
        train_loss += loss.data.item()
        pred = output.data.max(1)[1]
        total += label.size(0)
        iteration += 1
        train_error += pred.ne(label.data).cpu().sum()
    
    # Calculate Average loss & error rate per GPU        
    if args.flgDist:
        train_loss = torch.Tensor([train_loss]).cuda(args.idGPU)
        train_error = torch.Tensor([train_error]).cuda(args.idGPU)
        reject_counter = torch.Tensor([reject_counter]).cuda(args.idGPU)
        perturbation_norm = torch.Tensor([perturbation_norm]).cuda(args.idGPU)

        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(reject_counter, op=dist.ReduceOp.SUM)
        dist.all_reduce(perturbation_norm, op=dist.ReduceOp.SUM)

        train_loss = train_loss.item() / args.world_size
        train_error = train_error.item() / args.world_size
        reject_counter = reject_counter.item() / args.world_size
        perturbation_norm = perturbation_norm.item() / args.world_size

    train_loss = train_loss / iteration
    train_error = 100 * train_error / (iteration * args.train_batch_size_per_gpu)
    reject = 100 * reject_counter / iteration
    perturbation_norm = perturbation_norm / (iteration-reject_counter)
    
    return train_loss, train_error, reject, perturbation_norm
        

def main(args):
    global best_error

    # Set DataParallel Arguments
    if args.flgDist:
        master_addr = os.getenv("MASTER_ADDR", default="localhost")
        master_port = os.getenv('MASTER_PORT', default='8888')
        method = "tcp://{}:{}".format(master_addr, master_port)
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0')) # global addres of GPU
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1')) # number of total GPU
        ngpus_per_node = torch.cuda.device_count() # ngpus_per_node = 4
        args.idGPU = args.rank % ngpus_per_node
        print(f'rank : {args.rank}    world_size : {args.world_size}')
        torch.cuda.set_device(args.idGPU)
        dist.init_process_group('nccl', init_method=method, world_size=args.world_size, rank=args.rank)
        args.train_batch_size_per_gpu = int(args.train_total_batch_size / args.world_size)
        args.test_batch_size_per_gpu = int(args.test_total_batch_size / args.world_size)
        args.weak_clf_batch_size = int(args.weak_clf_batch_size / args.world_size)
    
    # Fix Seed
    if args.flgFixSeed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # Set Transformers
    if args.dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        image_size = 32
    elif args.dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        image_size = 32
    elif args.dataset == 'svhn':
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        image_size = 32
    elif args.dataset == 'fashion_mnist':
        mean = [0.2859]
        std = [0.3530]
        image_size = 28
        args.input_channels=1
    
    train_form = transforms.Compose([
        transforms.RandomCrop(image_size, padding=args.padding),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if args.augment_type == "cutout":
        train_form.transforms.append(Cutout(n_holes=1, length=args.co_length))
    if args.augment_type == "aa":
        train_form = transforms.Compose([
            transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    test_form = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # dataset
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./../dataset/cifar10_data/train/",
            train=True, 
            transform=train_form,
            download=True
        )
        test_dataset = datasets.CIFAR10(
            root="./../dataset/cifar10_data/test/",
            train=False,
            transform=test_form,
            download=True
        )
        numClass=10
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./../dataset/cifar100_data/train/",
            train=True, 
            transform=train_form,
            download=True
        )
        test_dataset = datasets.CIFAR100(
            root="./../dataset/cifar100_data/test/",
            train=False,
            transform=test_form,
            download=True
        )
        numClass=100
    elif args.dataset == 'svhn':
        train_dataset = ConcatDataset(
            [datasets.SVHN(
            root="./../dataset/svhn/train/",
            split='train', 
            transform=train_form,
            download=True
            ),
            datasets.SVHN(
            root="./../dataset/svhn/extra/",
            split='extra', 
            transform=train_form,
            download=True
            )]
        )
        test_dataset = datasets.SVHN(
            root="./../dataset/svhn/test/",
            split='test',
            transform=test_form,
            download=True
        )
        numClass=10
    elif args.dataset == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(
            root="./../dataset/fashion_mnist/train/",
            train=True, 
            transform=train_form,
            download=True
        )
        test_dataset = datasets.FashionMNIST(
            root="./../dataset/fashion_mnist/test/",
            train=False,
            transform=test_form,
            download=True
        )
        numClass=10
    
    # Set Sampler
    if args.flgDist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    else:
        train_sampler = None
        test_sampler = None
    
    # Set DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size_per_gpu,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler)

    # DataLoader for PoF
    if args.flgPoF:
        train_loader_forPoF = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.weak_clf_batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler)
    else:
        train_loader_forPoF = None

    # Set Network, optimizer, and loss
    net, optimizer = set_network(args, numClass) # args gets changed!!
    criterion = nn.CrossEntropyLoss().cuda(args.idGPU)

    if args.rank == 0:
        result = print_config(args, train_loader, net['numParam'])
    

    # Train
    else:
        training_time = 0
        for idxEpoch in range(args.start_epoch, args.end_epoch):
            start_time = time.perf_counter()
            if args.flgDist:
                train_sampler.set_epoch(idxEpoch)
            adjust_learning_rate(args, optimizer, idxEpoch)

            train_loss, train_error, reject, perturbation_norm = train(train_loader,train_loader_forPoF, 
                                                                        net, criterion, optimizer, args)

            if args.rank == 0:
                epoch_time = time.perf_counter()-start_time
                training_time = training_time + epoch_time
                print('Epoch: {:.2f}\tLoss: {:.4f}\tError: {:.2f}%\tlr_fet: {:.0e}\tlr_clf: {:.0e}\tReject: {:.2f}%\tperturb_norm: {:.4f}\t{:.2f}s/iter'.
                    format(idxEpoch+1, train_loss, train_error, optimizer['fet'].param_groups[0]['lr'], optimizer['clf'].param_groups[0]['lr'], reject, perturbation_norm, epoch_time), 
                    flush=True)
                result['train'].write('{},{:.6f},{}\n'.format(idxEpoch+1, train_loss, train_error))
                result['train'].flush()
            
            # Test
            if (idxEpoch+1) % args.freqValid == 0:
                if args.flgPoF:
                    if args.flgDist:
                        net['clf'].module.classifiers[-1].weight.data = net['weight'].clone()
                        net['clf'].module.classifiers[-1].bias.data = net['bias'].clone()
                    else:
                        net['clf'].classifiers[-1].weight.data = net['weight'].clone()
                        net['clf'].classifiers[-1].bias.data = net['bias'].clone()
                test_loss, test_error = test(test_loader, net, criterion, args)

                if args.rank == 0:
                    print('Test Epoch: {:.2f} \tloss: {:.4f} \tError: {:.2f}%\n'.format(idxEpoch+1, test_loss, test_error), flush=True)
                    result['test'].write('{},{:.6f},{}\n'.format(idxEpoch+1, test_loss, test_error))
                    result['test'].flush()
                is_best = test_error < best_error
                best_error = min(test_error, best_error)

            # Save Model
            if args.flgSaveModel:
                if ((idxEpoch+1)%args.freqSave==0) and (args.rank==0):
                    torch.save({'param': net['fet'].state_dict(), 'optim': optimizer['fet'].state_dict()}, os.path.join(args.dirSave, 'fet_{:d}.pth'.format(idxEpoch+1)))
                    if not args.flgPoF:
                        torch.save({'param': net['clf'].state_dict(), 'optim': optimizer['clf'].state_dict()}, os.path.join(args.dirSave, 'clf_{:d}.pth'.format(idxEpoch+1)))
        
        if args.flgDist:
            dist.barrier()
        if args.rank == 0:
            print('---------DONE---------')
            print('BEST_Test_Error_Rate: {} %'.format(best_error))
            result['train'].close()
            result['test'].close()

if __name__=='__main__':
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    args = set_args()
    main(args)

