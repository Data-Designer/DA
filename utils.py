#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 15:16
# @Author  : Jack Zhao
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# #Desc: tools
import os
import torch
import random
import csv
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import tqdm
########################################################
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from typing import Optional


from scipy.spatial.distance import cdist
from dataloader.cvloader import CVDataLoader
from config import opt
from torch.autograd import grad
from dataloader import office31

from torchvision import transforms
from dataloader.folder import ImageFolder_ind 

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def office31_load(args):
    train_bs = args.batch_size
    source = args.trans.split("2")[0]
    target = args.trans.split("2")[1]

    dset_loaders = {}
    dset_loaders["source_train"] = office31.get_office_dataloader(source, train_bs, True)
    dset_loaders["source_test"] = office31.get_office_dataloader(source, train_bs, True)
    dset_loaders["target_train"] = office31.get_office_dataloader(target, train_bs, True)
    dset_loaders["target_train_no_shuff"] = office31.get_office_dataloader(target, train_bs, False)
    dset_loaders["target_test"] = office31.get_office_dataloader(target, train_bs, True)

    return dset_loaders



def data_loader(train_path, val_path):
    """返回data_loader,我们不需要list文件,只需要按类别排序即可"""
    if opt.DATA == "clef" or opt.DATA == "office31" or opt.DATA == "modernoffice" or opt.DATA == "domainnet" or opt.DATA=='officecal'or opt.DATA=='officehome':
        data_transforms = {
            train_path: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224), # clef
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            val_path: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224), # clef
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
    elif opt.DATA == "visda":
        data_transforms = {
            train_path: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            val_path: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
    elif opt.DATA == "digit" and (opt.TRAPATH.split('/')[-1]== 'usps' or opt.VALPATH.split('/')[-1]== 'usps'):
        data_transforms = {
            train_path: transforms.Compose([
            transforms.Resize((28, 28)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]),

            val_path: transforms.Compose([ 
            transforms.Resize((28, 28)), # usps
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]),
        }
        
    elif opt.DATA == "digit" and (opt.TRAPATH.split('/')[-1]== 'svhn' or opt.VALPATH.split('/')[-1]== 'svhn'):
        data_transforms = {
            train_path: transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

            val_path: transforms.Compose([ 
            transforms.Resize((32, 32)), # usps
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])]),
        }
        
    elif opt.DATA == "cifa": 
        data_transforms = {
            train_path: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(size=(224,224)), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            val_path: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(size=(224,224)), 
#                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        }
        
    # dataset
    dsets = {x: ImageFolder_ind(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
    dsets_tgt_no_shuffle = ImageFolder_ind(os.path.join(val_path), data_transforms[val_path]) 

    # dataloader
    data_loader_T_no_shuffle = torch.utils.data.DataLoader(
        dsets_tgt_no_shuffle,
        batch_size=opt.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4)


    dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]} 
    dset_classes = dsets[train_path].classes
    print(dset_classes)
    classes_acc = {} 
    for i in dset_classes:
        classes_acc[i] = []
        classes_acc[i].append(0)
        classes_acc[i].append(0)

    train_loader = CVDataLoader()
    train_loader.initialize(dsets[train_path], dsets[val_path], opt.BATCH_SIZE, shuffle=True, drop_last=True)
    dataset = train_loader.load_data() 
    test_loader = CVDataLoader()
    test_loader.initialize(dsets[train_path], dsets[val_path], opt.BATCH_SIZE, shuffle=False, drop_last=False)
    dataset_test = test_loader.load_data()

    return dataset, dataset_test, data_loader_T_no_shuffle, classes_acc, dset_sizes, dset_classes, test_loader


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def weights_init(m):
    """初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def obtain_global_label(loader, model, glob_fea, glob_cls):
    """获取全局的伪标签，改动为K-NN"""
    start_test = True
    model.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs, labels, indexs = data[0], data[1], data[2]  # feature,label,

            inputs = inputs.cuda() 
            # if opt.DATA == 'digit' and (opt.TRAPATH.split('/')[-1]== 'usps' or opt.VALPATH.split('/')[-1]== 'usps'):
            #     inputs = inputs.mean(dim=1, keepdim=True)
            
            feas,outputs1,outputs2 = model(inputs)

            outputs = outputs1 + outputs2  # B,C(p)

            if start_test:  
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)  # softmax Dataset ,C
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])  

     
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)  # Dataset,H-> dataset,H+1
        
        
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        sample_sim = torch.mm(all_fea, all_fea.transpose(0,1)) # sample-smi, dataset * dataset
        sample_sim = (sample_sim.t() / torch.norm(sample_sim, p=2, dim=1)).t()
        
        topk, indices = torch.topk(sample_sim, opt.NEARK)
        weight = torch.zeros_like(sample_sim,device=sample_sim.device)
        del sample_sim
        torch.cuda.empty_cache()
        weight = weight.scatter(1, indices, topk)
        pred_generator = weight.mm(all_fea) # weight

        pred_logit = weight.mm(all_output) # weight
        weight, pred_label = torch.max(pred_logit, 1) # weight_
        pred_label = pred_label.float().cpu().numpy()
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy * 100, acc * 100)
        print(log_str + '\n')
        return weight, pred_label.astype('int'), pred_generator, accuracy * 100, acc * 100

    
    
    

def obtain_global_label_mem(data_t, index_t, model, glob_fea, glob_cls):
    """mem-bank"""
    start_test = True
    model.eval()
    with torch.no_grad():
        inputs = data_t
        feas,outputs1,outputs2 = model(inputs)

        outputs = outputs1 + outputs2  # B,C(p) 
        
        # outputs = nn.Softmax(dim=1)(outputs)  # 一次softmax Dataset ,C

        
        dis = torch.mm(feas.detach(), glob_fea.t())

        max_values, _ = torch.max(dis, dim=1, keepdim=True)
        dis.scatter_(1, index_t.view(-1, 1).to(max_values.device), -max_values)
        _, p1 = torch.sort(dis, dim=1)  # indice， B,Dataset
        
        # Create a mask for the top k indices
        mask = p1[:, :opt.NEARK]
        # Create a one-hot encoding for the top k indices
        one_hot = torch.zeros_like(dis).to(dis.device)
        one_hot.scatter_(1, mask, 1)
        # Calculate the weight for each row
        weight = one_hot / opt.NEARK
        
        pred_logit = weight.mm(glob_cls) # B,C

        weight_, pred_label = torch.max(pred_logit, 1)
        pred_label = pred_label.float().cpu().numpy()

        return weight_, pred_label.astype('int'), pred_logit


    
def chunk_mat(A, B, number_of_gpus):
    B_split = torch.split(B, B.shape[1]//number_of_gpus,dim=1)
    cos_split = []
    for i in range(number_of_gpus):
        device = 'cuda:{:d}'.format(i)
        cos_split.append(A.to(device) @ B_split[i].to(device))

    # DO THIS ONLY IF YOU HAVE ENOUGH CPU MEMORY!! :
    cos_split = torch.cat([d for d in D_split],dim=1)
    return cos_split


def discrepancy(out1, out2):
    """maintain diversity,"""
    return torch.mean(torch.abs(out1 - out2))




def gradient_discrepancy_evi_new(epoch, output, preds_s1, preds_s2, index_s1, index_s2, src_y, preds_t1, preds_t2, tgt_y, model):
    loss = evidential_criterion
    loss_w = Weighted_CrossEntropy

    c_candidate = list(range(opt.CLASSNUM))
    random.shuffle(c_candidate)
    total_loss = 0
    for c in c_candidate[0:opt.GRADNUM]:
        gm_loss = 0 
        src_ind = (src_y == c).nonzero().squeeze()  
        src_nind = (src_y != c).nonzero().squeeze() 
        tgt_ind = (tgt_y == c).nonzero().squeeze()
        if src_ind.shape == torch.Size([]) or tgt_ind.shape == torch.Size([]) or src_ind.shape == torch.Size(
                [0]) or tgt_ind.shape == torch.Size([0]) or src_nind.shape==torch.Size([]) or src_nind.shape == torch.Size(
                [0]):
            continue

        # output index
        p_s1, p_s2 = preds_s1[src_ind], preds_s2[src_ind]
        mask_s1, mask_s2 = index_s1[src_ind], index_s2[src_ind] 
        p_ns1, p_ns2 = preds_s1[src_nind], preds_s2[src_nind]
        mask_ns1, mask_ns2 = index_s1[src_nind], index_s2[src_nind] 
        p_t1, p_t2 = preds_t1[tgt_ind], preds_t2[tgt_ind]

        s_y, s_ny, t_y = src_y[src_ind], src_y[src_nind], tgt_y[tgt_ind] 
        
        
        # visda
        # src_loss1, tgt_loss1 = loss(p_s1, one_hot_embedding(s_y, opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10,
                                    # p_s1.device).mean(), loss(p_t1, one_hot_embedding(t_y, opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10,
                                    # p_t1.device).mean()
        # src_loss2, tgt_loss2 = loss(p_s2, one_hot_embedding(s_y, opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10,
                                    # p_s2.device).mean(), loss(p_t2, one_hot_embedding(t_y, opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10,
                                    # p_t2.device).mean()
        
        # loss
        src_loss1, tgt_loss1 = nn.CrossEntropyLoss(reduction='none')(p_s1,s_y), loss_w(p_t1,t_y)
        src_loss2, tgt_loss2 = nn.CrossEntropyLoss(reduction='none')(p_s1,s_y), loss_w(p_t2,t_y)
        src_loss1, src_loss2 = src_loss1.mul(mask_s1).mean(), src_loss2.mul(mask_s2).mean()

        
        # grad
        grad_src = torch.autograd.grad(outputs=src_loss1+src_loss2, inputs=output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_tgt = torch.autograd.grad(outputs=tgt_loss1+tgt_loss2, inputs=output, create_graph=True, retain_graph=True, only_inputs=True)[0]

        _cossim = F.cosine_similarity(grad_src.detach(), grad_tgt.detach(), dim=1).mean()
        gm_loss = (1.0 - _cossim) 
        total_loss += gm_loss
        
    if total_loss == 0:
        return 0

    return (total_loss / opt.GRADNUM)



def uncertainty_prob(preds):
    alpha = relu_evidence(preds) + 1  # B,C
    u_s1 = opt.CLASSNUM / torch.sum(alpha, dim=1, keepdim=True)  # B,1
    return u_s1


def uncertainty(preds):
    alpha = relu_evidence(preds) + 1  # B,C
    u_s1 = opt.CLASSNUM / torch.sum(alpha, dim=1, keepdim=True)  # B,1

    """visda"""
    value, index = torch.kthvalue(u_s1.view(-1,), round(opt.BATCH_SIZE * opt.UNCERTAINTY)) 
    index = (u_s1 < value) 

  
    return index


def relu_evidence(y):
    # return y
    # return F.relu(y)
    return torch.exp(torch.clamp(y, -10, 10))


def kl_divergence(alpha, num_class, device):
    uni = torch.ones([1, num_class], dtype=torch.float32,device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(uni).sum(dim=1, keepdim=True)
            - torch.lgamma(uni.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - uni)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl_loss = first_term + second_term
    return kl_loss


def evidential_loss(func, target, alpha, epoch_num, num_classes, annealing_step, device=None):
    target = target.to(device)
    # alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True) # B,1
    A = torch.sum(target * (func(S) - func(alpha)), dim=1, keepdim=True) # B,C
    # 正则化
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - target) + 1 
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)

    # print(A)
    # print("KL",kl_div)

    return A + kl_div * 0.005


def evidential_criterion(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output) # B,T
    alpha = evidence + 1
    loss = evidential_loss(torch.digamma,
                                      target, # B,C
                                      alpha, epoch_num,
                                      num_classes, annealing_step, device) # torch.mean

    return loss



def probality_estimate(output):
    evidence = relu_evidence(output) # B,C
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True) # B,1
    return alpha/S


def Entropy_div(input_):
    # B,C
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon # 1,C
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy # value

def Entropy_condition(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5) # B,C
    entropy = torch.sum(entropy, dim=1).mean() # B每个样本熵的平均
    return entropy

def Entropy(input_):
    return Entropy_condition(input_) + Entropy_div(input_)


def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_) # B,C
    entropy = -input_s * torch.log(input_s + 1e-5) # B,C
    entropy = torch.sum(entropy, dim=1) # B
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item() #

    weight_result = weight * nn.CrossEntropyLoss(reduction='none')(input_, labels) # B, 1
    return torch.mean(weight_result)


def write_csv(results, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['gameplayid', 'h_local_prob','h_pred_glob','a_local_prob','a_pred_glob',"h_batch_attn","a_batch_attn"])
        writer.writerows(results)


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct



class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
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
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x



def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance



from itertools import chain
class DataParallelFix(nn.DataParallel):
    """
    Temporary workaround for https://github.com/pytorch/pytorch/issues/15716.
    """
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
        self._replicas = []
        self._outputs = []
        self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))
 
    def reset(self):
        self._replicas = []
        self._outputs = []
 
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
 
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj,
                                                t.device))
 
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
 
        _replicas = self.replicate(self.module,
                                  self.device_ids[:len(inputs)])
        _outputs = self.parallel_apply(_replicas, inputs, kwargs)
        self._replicas.append(_replicas)
        self._outputs.append(_outputs)
        return self.gather(_outputs, self.output_device)