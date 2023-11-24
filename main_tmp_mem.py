#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 15:17
# @Author  : Jack Zhao
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc: memory bank

import torch
import fire
import datetime
import warnings
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from draw.vis import visual_tsne
from config import opt
from models.base import *
from utils import *
# from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd.variable import Variable


warnings.filterwarnings('ignore')
torch.manual_seed(opt.SEED)
if opt.GPU_USED:
    torch.cuda.manual_seed(opt.SEED)
# seed_everything(opt.SEED)
writer = SummaryWriter(opt.TLOGFILE) 




def case(**kwargs):
    """
    其实就是测试,做case study
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)

    option = 'resnet' + str(opt.RESNET)
    G = GeneratorRes(option)
    dim = G.output_dim()
    F1 = Discriminator(num_cls=opt.CLASSNUM, num_layer=opt.DISLAYER, num_unit=dim, middle=1000)
    F2 = Discriminator(num_cls=opt.CLASSNUM, num_layer=opt.DISLAYER, num_unit=dim, middle=1000)

    results = [] 

    # opt
    pth = os.listdir(opt.WEIGHTS)
    pth_f1 = [i for i in pth if i.startswith('F1')]
    pth_f2 = [i for i in pth if i.startswith('F2')]
    pth_g = [i for i in pth if i.startswith('G')]
    pth_f1.sort(key=lambda x: int(x.split('_')[1][:-4])),pth_f2.sort(key=lambda x: int(x.split('_')[1][:-4])),pth_g.sort(key=lambda x: int(x.split('_')[1][:-4]))
    f1_best,f2_best,g_best = pth_f1[-1],pth_f2[-1],pth_g[-1] # load
    f1_dict,f2_dict,g_dict = torch.load(opt.WEIGHTS + f1_best),torch.load(opt.WEIGHTS + f2_best),torch.load(opt.WEIGHTS + g_best)
    F1.load_state_dict(f1_dict),F2.load_state_dict(f2_dict),G.load_state_dict(g_dict)  

    if opt.GPU_USED:
        G, F1, F2 = G.cuda(), F1.cuda(), F2.cuda()

    print("Start Casing...")
    # load
    dataset, dataset_test, data_loader_T_no_shuffle, classes_acc, dset_sizes, dset_classes, test_loader = data_loader(opt.TRAPATH, opt.VALPATH)

    G.eval()
    F1.eval()
    F2.eval()
    case_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    for batch_idx, data in enumerate(dataset_test):
        if dataset_test.stop_T:
            break
        img, label, index = data['T'], data['T_label'], data['T_index']
        if opt.GPU_USED:
            img, label = data['T'].cuda(), data['T_label'].cuda()
        img, label = Variable(img, volatile=True), Variable(label,volatile=True) 
        with torch.no_grad():
            output = G(img)
            output_1 = F1(output)
            output_2 = F2(output)

        case_loss += F.nll_loss(output, label).item() 
        output_add = output_1 + output_2
        pred = output_add.data.max(dim=1)[1] # indice,label
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]

        for i in range(len(label)): 
            key_label = dset_classes[label.long()[i].item()]
            key_pred = dset_classes[pred.long()[i].item()]
            classes_acc[key_label][1] += 1 
            if key_pred == key_label:
                classes_acc[key_pred][0] += 1
    case_loss = case_loss / len(test_loader) # batch loss have been averages
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
         case_loss, correct_add, size, 100. * float(correct_add) / size))

    avg = []
    for i in dset_classes: 
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    cls_avg_acc = np.average(avg)
    print('\tClass accuracy average:', cls_avg_acc) 
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0
    print("Finish CASING...")

    print("Start Visualizing !")
    all_feature_source = []
    all_feature_target = []
    for batch_idx, data in enumerate(dataset):
        data_s, label_s, data_t, label_t, index_t = data['S'], data['S_label'], data['T'], data['T_label'], data[
            'T_index']
        if opt.GPU_USED:
            data_s, label_s = data_s.cuda(), label_s.cuda()  # B,C,W,H; B;
            data_t, label_t = data_t.cuda(), label_t.cuda()
        data_all = Variable(torch.cat((data_s, data_t), 0),volatile=True)  # 2B,H, 
        label_s = Variable(label_s,volatile=True)

        bs = len(label_s)
        with torch.no_grad():
            # Step A train all networks to minimize loss on source
            output = G(data_all)
            output_1,output_2 = output,output
            # output_1 = F1(output)
            # output_2 = F2(output)

            output_s1, output_s2 = output_1[:bs, :], output_2[:bs, :]
            output_t1, output_t2 = output_1[bs:, :], output_2[bs:, :]
            source = F.softmax(output_s1 + output_s2).cpu()
            target = F.softmax(output_t1 + output_t2).cpu() # 归一化
            all_feature_source.append(source)
            all_feature_target.append(target)

    all_feature_source = torch.cat(all_feature_source, dim=0)
    all_feature_target = torch.cat(all_feature_target, dim=0)

    visual_tsne(all_feature_source, all_feature_target, opt.TSNE, source_color='r', target_color='g')
    print("Finish Visualizing !")

    print("Calculate A-distance")
    device = 'cpu'
    if opt.GPU_USED:
        device = 'cuda:0'
    A_distance = calculate(all_feature_source, all_feature_target, device)
    print("A-distance =", A_distance)
    print("All Finish")

    return case_loss, 100. * float(correct_add) / size, avg, cls_avg_acc



def train(**kwargs):
    """train"""
    opt.parse(kwargs)


    device = 'cpu'
    use_cuda = opt.GPU_USED
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
        
        
    dataset, dataset_test, data_loader_T_no_shuffle, classes_acc, dset_sizes, dset_classes, test_loader = data_loader(opt.TRAPATH, opt.VALPATH)

    # model definition
    if opt.DATA == 'digit' and (opt.TRAPATH.split('/')[-1]== 'usps' or opt.VALPATH.split('/')[-1]== 'usps'):
        option="lenet" # lenet
    elif opt.DATA == 'digit' and (opt.TRAPATH.split('/')[-1]== 'svhn' or opt.VALPATH.split('/')[-1]== 'svhn'):
        option="dtn"
#     elif opt.DATA == 'cifa':
#         option="lenet" # lenet
    else:
        option='resnet' + str(opt.RESNET)
        
    print(option)
    
    # model
    model = BaseNetwork(option=option, pretrain=True, use_bottleneck=opt.USE_BOTTLE, device = device)
    g_params = model.get_generator_learner_parameters()
    f_params = model.get_classifier_parameters()

    global_fea = torch.rand(len(data_loader_T_no_shuffle.dataset), model.g_dim).to(device) # Dataset,H(C)
    global_fea_norm = global_fea / torch.norm(global_fea, p=2, dim=1, keepdim=True) # Norm
    global_cls = torch.ones(len(data_loader_T_no_shuffle.dataset), len(dset_classes)).to(device) / len(dset_classes)  # Dataset, C

    # opt
    if opt.OPTIM == 'momentum':  #
        optimizer_g = optim.SGD(g_params, lr=opt.LR,
                                weight_decay=opt.WEIGHT_DECAY)  
        optimizer_f = optim.SGD(f_params, momentum=0.9, lr=opt.LR,
                                weight_decay=opt.WEIGHT_DECAY)
    elif opt.OPTIM == 'adam':
        optimizer_g = optim.Adam(g_params, lr=opt.LR, weight_decay=opt.WEIGHT_DECAY)
        optimizer_f = optim.Adam(f_params, lr=opt.LR,
                                 weight_decay=opt.WEIGHT_DECAY)
    else:
        optimizer_g = optim.Adadelta(g_params, lr=opt.LR, weight_decay=opt.WEIGHT_DECAY)
        optimizer_f = optim.Adadelta(f_params, lr=opt.LR,
                                     weight_decay=opt.WEIGHT_DECAY)

    criterion = evidential_criterion
    criterion_w = Weighted_CrossEntropy

    dfhistory = pd.DataFrame(columns=['epoch','test_loss','test_acc','avg','best_acc','cls_avg_acc',"time"])

    best_acc = 0.0

    # train
    print("Start Training !")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%now_time)
    start = opt.START
    iters = 0
    for epoch in range(1,opt.EPOCHS+1):

        # train step
        for batch_idx, data in enumerate(dataset):
     
            import time
            start_times = time.time()
            iters += 1

            target_conf = 0.1
            data_s, label_s, data_t, label_t, index_t = data['S'], data['S_label'], data['T'], data['T_label'], data[
                'T_index']
            data_s, label_s = data_s.to(device), label_s.to(device) # B,C,W,H; B;
            data_t, label_t = data_t.to(device), label_t.to(device)
            
            # if opt.DATA == 'digit' and (opt.TRAPATH.split('/')[-1]== 'usps' or opt.VALPATH.split('/')[-1]== 'usps'):
                # data_s,data_t = data_s.mean(dim=1, keepdim=True), data_t.mean(dim=1, keepdim=True)
            
            if epoch > start:
                # print("Obtaining the target domain label...")
                model.eval()
                weight_, pseudo_label_t, pseudo_pred_logit = obtain_global_label_mem(data_t.clone(), index_t, model, global_fea_norm, global_cls) # 改进KNN
                pseudo_label_t = torch.from_numpy(pseudo_label_t).to(device)
                weight_ = weight_.to(device) 

            model.train()
            if dataset.stop_S:
                break
            
            c_loss = 0
            if epoch > start:
                # 
                label_set = set(pseudo_label_t.cpu().long().view(-1).numpy().tolist())
                pseudo_label_t = pseudo_label_t.to(device)
                pseudo_pred_logit = pseudo_pred_logit.to(device) # 
                pseudo_pred_logit = pseudo_pred_logit
                c_loss = 0
                for c in label_set:
                    tgt_ind = (pseudo_label_t == c).nonzero().squeeze()  
                    tgt_nind = (pseudo_label_t != c).nonzero().squeeze()  
                    # visda 3
                    if tgt_ind.shape == torch.Size([]) or tgt_nind.shape == torch.Size([]) or tgt_ind.shape[0]<4 or tgt_nind.shape[0]<4: 
                        continue
                    tgt_sim_logit, tgt_diff_logit = pseudo_pred_logit[tgt_ind], pseudo_pred_logit[tgt_nind] # K1,C; K2,C
                    tgt_sim, tgt_diff = torch.mm(tgt_sim_logit, tgt_sim_logit.transpose(0,1)), torch.mm(tgt_sim_logit, tgt_diff_logit.transpose(0,1)) # K1,K1;K1,K2
                    tgt_sim, tgt_diff = (tgt_sim.t() / torch.norm(tgt_sim, p=2, dim=1)).t().abs(), (tgt_diff.t() / torch.norm(tgt_diff, p=2, dim=1)).t().abs() # visda修改
                    # tgt_sim, tgt_diff = (tgt_sim.t() / torch.norm(tgt_sim, p=2, dim=1)).t(), (tgt_diff.t() / torch.norm(tgt_diff, p=2, dim=1)).t()

                    weight_smi, weight_diff = (tgt_sim > torch.topk(tgt_sim, 4)[0][..., -1, None]).float(),(tgt_diff > torch.topk(tgt_diff, 4)[0][..., -1, None]).float() # 最近邻2个的index
                    sim_logit, diff_logit = weight_smi.mm(tgt_sim_logit), weight_diff.mm(tgt_diff_logit) # 2,C;2,C
                    imp_knn_loss = max(torch.mean(sim_logit) + 0.00005 - torch.mean(diff_logit), 0) # sim小于diff，并且最小差gap a
                    c_loss += imp_knn_loss
                c_loss = c_loss/(len(label_set))

            data_all = Variable(torch.cat((data_s, data_t), 0))  # 2B,H, 方便后期统一过模型
            label_s = Variable(label_s)
            bs = len(label_s)

            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output,output_1,output_2 = model(data_all)

            output_s1, output_s2 = output_1[:bs, :], output_2[:bs, :]
            output_t1, output_t2 = output_1[bs:, :], output_2[bs:, :]
            output_t1_sx, output_t2_sx = F.softmax(output_t1), F.softmax(output_t2)

            entropy_loss = Entropy(output_t1_sx) + Entropy(output_t2_sx)

            if epoch > start:
                # ATDOC
                loss_ = nn.CrossEntropyLoss(reduction='none')(output_t1, pseudo_label_t) + nn.CrossEntropyLoss(reduction='none')(output_t2, pseudo_label_t)
                supervision_loss_t = torch.mean(weight_ * loss_) / (torch.sum(weight_).item())  # ATDOC loss

            else:
                supervision_loss_t = 0 

            # print(label_s.shape)
            loss1, loss2 = nn.CrossEntropyLoss(reduction='mean')(output_s1,label_s), nn.CrossEntropyLoss(reduction='mean')(output_s2,label_s)
            supervision_loss_s = loss1 + loss2


            all_loss = supervision_loss_s + opt.ENTROPY * entropy_loss + 0.1 * supervision_loss_t + opt.CLOSS * c_loss  
            # all_loss = supervision_loss_s + 0.1 * entropy_loss + 0.1 * supervision_loss_t
            # print(supervision_loss_s, entropy_loss, supervision_loss_t, c_loss)

            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            
            
            # Step B train classifiers to maximize CDD loss, main target diversity
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output,output_1,output_2 = model(data_all)

            output_s1, output_s2 = output_1[:bs, :], output_2[:bs, :]
            output_t1, output_t2 = output_1[bs:, :], output_2[bs:, :]
            output_t1_sx, output_t2_sx = F.softmax(output_t1), F.softmax(output_t2)
            entropy_loss = Entropy(output_t1_sx) + Entropy(output_t2_sx)
            loss1, loss2 = nn.CrossEntropyLoss(reduction='mean')(output_s1,label_s), nn.CrossEntropyLoss(reduction='mean')(output_s2,label_s)
            
            loss_disc = discrepancy(output_t1_sx, output_t2_sx)
            supervision_loss_s = loss1 + loss2

            all_loss = supervision_loss_s - opt.DISC * loss_disc + opt.ENTROPY * entropy_loss
            all_loss.backward()
            
            
            optimizer_f.step()

            # Step C train generator to minimize CDD loss, 
            for i in range(opt.GUPDATE):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                output,output_1,output_2 = model(data_all)
                output_s1, output_s2 = output_1[:bs, :], output_2[:bs, :]
                output_t1, output_t2 = output_1[bs:, :], output_2[bs:, :]
                output_t1_sx, output_t2_sx = F.softmax(output_t1), F.softmax(output_t2)

                entropy_loss = Entropy(output_t1_sx) + Entropy(output_t2_sx)
                loss_disc = discrepancy(output_t1_sx, output_t2_sx)

                index_s1 = uncertainty(output_s1)  # B,1
                index_s2 = uncertainty(output_s2)

                if epoch > start:
                    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    # gmn_loss = gradient_discrepancy(output_s1, output_s2, label_s, output_t1, output_t2,
                    #                                     pseudo_label_t, G, F1, F2)
                    gmn_loss = gradient_discrepancy_evi_new(epoch, output, output_s1, output_s2, index_s1, index_s2, label_s, output_t1, output_t2,
                                                         pseudo_label_t, model)
                    # print(prof)
                else:
                    gmn_loss = 0
                all_loss = opt.DISC * loss_disc + opt.ENTROPY * entropy_loss + opt.GMLOSS * gmn_loss

                all_loss.backward()
                optimizer_g.step()

            # 更新mem bank
            with torch.no_grad():
                model.eval()
                data = data_t
                feas,output_1,output_2 = model(data)
                outputs = output_1 + output_2  # B,C(p)
                features_target = feas / torch.norm(feas, p=2, dim=1, keepdim=True)
                softmax_out = nn.Softmax(dim=1)(outputs)
                outputs_target = softmax_out
                model.train()
                # outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))
            # print(global_fea)
            # print(global_fea.shape, features_target.shape, global_cls.shape,outputs_target.shape)
            # print(index_t.shape)
            if epoch ==1:
                global_fea[index_t] = features_target.clone()
                global_cls[index_t] = outputs_target.clone()
            else:
                global_fea[index_t] = (1.0 - opt.MEMENTUM) * global_fea[index_t] + opt.MEMENTUM * features_target.clone()
                global_cls[index_t] = (1.0 - opt.MEMENTUM) * global_cls[index_t] + opt.MEMENTUM * outputs_target.clone()

            if batch_idx % opt.LOG_STEP_FREQ == 0:
                # CDD
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t CDD: {:.6f} Entropy: {:.6f} '.format(
                        epoch, batch_idx, len(dataset.data_loader_S), 100. * batch_idx / len(dataset.data_loader_S),
                        loss1.item(), loss2.item(), loss_disc.item(), entropy_loss.item()))
            end_times = time.time()
            # print("One step time, ", end_times-start_times)
        # test
        test_loss, test_acc, avg, cls_avg_acc = test(epoch, model, dataset_test, dset_classes, classes_acc, test_loader, device)
        

        if test_acc > best_acc:
            torch.save(model.state_dict(), opt.WEIGHTS + "BEST_{}.pth".format(epoch))
            best_acc = test_acc
        print("best acc:", best_acc)

        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("==========" * 8 + "%s\n" % (end_time))

        # save
        times = (datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(now_time,
                                                                                                       '%Y-%m-%d %H:%M:%S')).seconds
        
        # print("RUN time", time)
        info = (epoch, test_loss, test_acc, avg, best_acc, cls_avg_acc, time)
        dfhistory.loc[epoch - 1] = info



    print("Finished Training...")
    dfhistory.to_csv(opt.LOGFILE)
    print("LOG FILE HAVE SAVED!")


def test(epoch, model, dataset_test, dset_classes, classes_acc,test_loader, device):
    """测试阶段, return float,float,obj,float"""
    model.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    for batch_idx, data in enumerate(dataset_test):
        if dataset_test.stop_T:
            break
        img, label, index = data['T'].to(device), data['T_label'].to(device), data['T_index']
        img, label = Variable(img, volatile=True), Variable(label) # 防止embed反向传播
        
        # if opt.DATA == 'digit' and (opt.TRAPATH.split('/')[-1]== 'usps' or opt.VALPATH.split('/')[-1]== 'usps'):
            # img = img.mean(dim=1, keepdim=True)
        
        with torch.no_grad():
            output,output_1,output_2 = model(img)
            test_output = (output_1 + output_2)/2


        test_loss += F.nll_loss(test_output, label).item() 
        output_add = output_1 + output_2
        pred = output_add.data.max(dim=1)[1] # indice,label
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]

        for i in range(len(label)): 

            key_label = dset_classes[label.long()[i].item()]
            key_pred = dset_classes[pred.long()[i].item()]
            classes_acc[key_label][1] += 1 
            if key_pred == key_label:
                classes_acc[key_pred][0] += 1
    test_loss = test_loss / len(test_loader) # batch loss have been averages
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))

    avg = []
    for i in dset_classes: #
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    cls_acc = classes_acc
    cls_avg_acc = np.average(avg)
    print('\tClass accuracy average:', cls_avg_acc) #
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0

    return test_loss, 100. * float(correct_add) / size, avg, cls_avg_acc





if __name__ == '__main__':
    # fire.Fire()
    train()
    # case()