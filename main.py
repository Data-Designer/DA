#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 15:17
# @Author  : Jack Zhao
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc: main

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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd.variable import Variable


warnings.filterwarnings('ignore')
torch.manual_seed(opt.SEED)
# if opt.GPU_USED:
#     torch.cuda.manual_seed(opt.SEED)
# writer = SummaryWriter(opt.TLOGFILE) # 定义Tensorboard句柄

# # seed_everything(opt.SEED)



def draw_ana():
  
    option = 'resnet' + str(101)
    feature_extrator = GeneratorRes(option).cuda()
    TRAPATH = "/dfs/data/DA/data/visda/train"
    VALPATH = "/dfs/data/DA/data/visda/validation"
    dataset, dataset_test, data_loader_T_no_shuffle, classes_acc, dset_sizes, dset_classes, test_loader = data_loader(
        TRAPATH, VALPATH)
    feature_extrator.eval()
    features = []
    labels = []
    features_t = []
    labels_t = []
    for batch_idx, data in enumerate(dataset):
        data_s, label_s, data_t, label_t, index_t = data['S'], data['S_label'], data['T'], data['T_label'], data[
            'T_index']
        data_s, label_s = data_s.cuda(), label_s.cuda()
        data_t, label_t = data_t.cuda(), label_t.cuda()
        feature = feature_extrator(data_s).cpu().detach()  
        feature_t = feature_extrator(data_t).cpu().detach()
        # print(feature.shape)
        features.append(feature)
        # labels.append(label_s)
        features_t.append(feature_t)
        # labels_t.append(label_t)
        # if batch_idx % 32000==0:
        #     print()
    features = torch.cat(features, dim=0)
    # labels = torch.cat(labels, dim=0)
    features_t = torch.cat(features_t, dim=0)
    # labels = torch.cat(labels_t, dim=0)
    len_sample = features.shape[0]//10

    visual_tsne(features[:len_sample,:], features_t[:len_sample,:], opt.TSNE, source_color='r', target_color='b')
    # print(features.shape)
    # plot_outlier(features)


def case(**kwargs):
    """
    case study
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)
    # 模型定义
    if opt.DATA == "digit":
        option="lenet"
    else:
        option='resnet' + str(opt.RESNET)
        
    model = BaseNetwork(option=option, pretrain=False, use_bottleneck=True, device = device)

    results = [] 

 
    pth = os.listdir(opt.WEIGHTS)
    pth.sort(key=lambda x: int(x.split('_')[1][:-4]))
    best_num = pth_f1[-1]
    model_dict = torch.load(opt.WEIGHTS + best_num)
    model.load_state_dict(model_dict)

    if opt.GPU_USED:
        model = model.cuda()

    print("Start Casing...")
    # load
    dataset, dataset_test, data_loader_T_no_shuffle, classes_acc, dset_sizes, dset_classes, test_loader = data_loader(opt.TRAPATH, opt.VALPATH)

    model.eval()
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
            output,output_1,output_2 = model(img)

        case_loss += F.nll_loss(output, label).item() # 
        output_add = output_1 + output_2
        pred = output_add.data.max(dim=1)[1] # indice,label
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]

        for i in range(len(label)): # 
            key_label = dset_classes[label.long()[i].item()]
            key_pred = dset_classes[pred.long()[i].item()]
            classes_acc[key_label][1] += 1 # 
            if key_pred == key_label:
                classes_acc[key_pred][0] += 1
    case_loss = case_loss / len(test_loader) # batch loss have been averages
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
         case_loss, correct_add, size, 100. * float(correct_add) / size))

    avg = []
    for i in dset_classes: # 
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    cls_avg_acc = np.average(avg)
    print('\tClass accuracy average:', cls_avg_acc) # 
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
            output,output_1,output_2 = model(data_all)

            output_s1, output_s2 = output_1[:bs, :], output_2[:bs, :]
            output_t1, output_t2 = output_1[bs:, :], output_2[bs:, :]
            source = F.softmax(output_s1 + output_s2).cpu()
            target = F.softmax(output_t1 + output_t2).cpu() # 归一化
            all_feature_source.append(source)
            all_feature_target.append(target)

    all_feature_source = torch.cat(all_feature_source, dim=0)
    all_feature_target = torch.cat(all_feature_target, dim=0)

    visual_tsne(all_feature_source, all_feature_target, opt.TSNE, source_color='r', target_color='b')
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
    
    # config definition & data definition
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
    print("option", option)
        
    model = BaseNetwork(option=option, pretrain=True, use_bottleneck=opt.USE_BOTTLE, device = device)
    g_params = model.get_generator_learner_parameters()
    f_params = model.get_classifier_parameters()
    
    gpus = 1 # opt.GPU_NUM
    if gpus > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in range(gpus)])
    

    if opt.OPTIM == 'momentum': 
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

    global_fea = torch.rand(len(data_loader_T_no_shuffle.dataset), len(dset_classes)).to(device) # Dataset,H(C)
    global_fea_norm = global_fea / torch.norm(global_fea, p=2, dim=1, keepdim=True) # Norm
    global_cls = torch.ones(len(data_loader_T_no_shuffle.dataset), len(dset_classes)).to(device) / len(dset_classes)  # Dataset, C


    criterion = evidential_criterion
    criterion_w = Weighted_CrossEntropy

    # storage definition
    dfhistory = pd.DataFrame(columns=['epoch','test_loss','test_acc','avg','best_acc','cls_avg_acc',"cluster_b","cluster_a","time_interval","train_loss"])

    best_acc = 0.0

    # train
    print("Start Training !")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%now_time)
    start = opt.START
    
    for epoch in range(1, opt.EPOCHS+1):
        iter_num = 0
        # train step
        train_weight = 0.1 + 0.3/opt.EPOCHS*epoch
        pre_acc, now_acc = 0, 0     
        
        for batch_idx, data in enumerate(dataset):
            import time
            start_times = time.time()
            
            iter_num += 1
            if epoch > start and batch_idx % opt.PSEUINTERVEAL == 0:  
                print("Obtaining the target domain label...")
                model.eval()
                weight_, mem_label, pred_logit, pre_acc, now_acc = obtain_global_label(data_loader_T_no_shuffle, model, global_fea_norm, global_cls) # 改进KNN
                
                mem_label = torch.from_numpy(mem_label).to(device)
                weight_ = weight_.to(device) 


            model.train()
            data_s, label_s, data_t, label_t, index_t = data['S'], data['S_label'], data['T'], data['T_label'], data[
                'T_index']
            if epoch > start:
                pseudo_label_t = mem_label[index_t]  
                pseudo_pred_logit = pred_logit[index_t] # B,H



            if dataset.stop_S: 
                break
                
            data_s, label_s = data_s.to(device), label_s.to(device) # B,C,W,H; B;
            data_t, label_t = data_t.to(device), label_t.to(device)
            
#             if opt.DATA == 'digit' and (opt.TRAPATH.split('/')[-1]== 'usps' or opt.VALPATH.split('/')[-1]== 'usps'):
#                 data_s,data_t = data_s.mean(dim=1, keepdim=True), data_t.mean(dim=1, keepdim=True)
                
            
            # KNN C-loss definition
            c_loss = 0
            if epoch > start:
                label_set = set(pseudo_label_t.cpu().long().view(-1).numpy().tolist())
                pseudo_label_t = pseudo_label_t.to(device)
                pseudo_pred_logit = pseudo_pred_logit.to(device) # 改进KNN
                pseudo_pred_logit = pseudo_pred_logit
                c_loss = 0
                for c in label_set:
                    tgt_ind = (pseudo_label_t == c).nonzero().squeeze()  # 找出每一类的index
                    tgt_nind = (pseudo_label_t != c).nonzero().squeeze()  # 其他类的index
                    # visda 3/4
                    if tgt_ind.shape == torch.Size([]) or tgt_nind.shape == torch.Size([]) or tgt_ind.shape[0]<2 or tgt_nind.shape[0]<2: 
                        continue
                    tgt_sim_logit, tgt_diff_logit = pseudo_pred_logit[tgt_ind], pseudo_pred_logit[tgt_nind] # K1,C; K2,C
                    tgt_sim, tgt_diff = torch.mm(tgt_sim_logit, tgt_sim_logit.transpose(0,1)), torch.mm(tgt_sim_logit, tgt_diff_logit.transpose(0,1)) # K1,K1;K1,K2
                    tgt_sim, tgt_diff = (tgt_sim.t() / torch.norm(tgt_sim, p=2, dim=1)).t(), (tgt_diff.t() / torch.norm(tgt_diff, p=2, dim=1)).t() # visda修改，修改不需要abs?, 影响比较大
                    # tgt_sim, tgt_diff = (tgt_sim.t() / torch.norm(tgt_sim, p=2, dim=1)).t(), (tgt_diff.t() / torch.norm(tgt_diff, p=2, dim=1)).t()

                    weight_smi, weight_diff = (tgt_sim > torch.topk(tgt_sim, 2)[0][..., -1, None]).float(),(tgt_diff > torch.topk(tgt_diff, 2)[0][..., -1, None]).float() # 最近邻2个的index
                    sim_logit, diff_logit = weight_smi.mm(tgt_sim_logit), weight_diff.mm(tgt_diff_logit) 
                    imp_knn_loss = max(torch.mean(sim_logit) + 0.00005 - torch.mean(diff_logit), 0) 
                    c_loss += imp_knn_loss
                c_loss = c_loss/(len(label_set))
                

            data_all = Variable(torch.cat((data_s, data_t), 0))  # 2B,H
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
            # entropy_loss = 0
            # output_s1, output_s2 = probality_estimate(output_s1), probality_estimate(output_s2)

            if epoch > start:                
                supervision_loss_t = torch.mean(criterion(output_t1, one_hot_embedding(pseudo_label_t,opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10, output_t1.device))+torch.mean(criterion(output_t2, one_hot_embedding(pseudo_label_t,opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10, output_t2.device))
            else:
                supervision_loss_t = 0 

            loss1, loss2 = torch.mean(criterion(output_s1, one_hot_embedding(label_s,opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10, output_s1.device)), \
                           torch.mean(criterion(output_s2, one_hot_embedding(label_s,opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10, output_s2.device)) # 这个10是不是要调整一下啊,source evi
            supervision_loss_s = loss1 + loss2
            
            
            all_loss = supervision_loss_s + opt.ENTROPY * entropy_loss + 0.1 * supervision_loss_t + opt.CLOSS * c_loss  

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
            
            # output_s1, output_s2 = probality_estimate(output_s1), probality_estimate(output_s2)

            loss1, loss2 = torch.mean(criterion(output_s1, one_hot_embedding(label_s,opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10, output_s1.device)), \
                           torch.mean(criterion(output_s2, one_hot_embedding(label_s,opt.CLASSNUM).float(), epoch, opt.CLASSNUM, 10, output_s2.device))
            # loss1, loss2 = nn.CrossEntropyLoss(reduction='mean')(output_s1,label_s), nn.CrossEntropyLoss(reduction='mean')(output_s2,label_s)
            entropy_loss = Entropy(output_t1_sx) + Entropy(output_t2_sx)

            supervision_loss_s = loss1 + loss2 
            loss_disc = discrepancy(output_t1_sx, output_t2_sx) 
            # entropy_loss = 0
            # loss_disc = discrepancy(probality_estimate(output_t1), probality_estimate(output_t2))

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
                    gmn_loss = gradient_discrepancy_evi_new(epoch, output, output_s1, output_s2, index_s1, index_s2, label_s, output_t1, output_t2,
                                                         pseudo_label_t, model)
                else:
                    gmn_loss = 0
                

                all_loss = opt.DISC * loss_disc + opt.ENTROPY * entropy_loss + opt.GMLOSS * gmn_loss


                all_loss.backward()
                optimizer_g.step()



            if batch_idx % opt.LOG_STEP_FREQ == 0: 
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t CDD: {:.6f} Entropy: {:.6f} CLoss: {:.6f}, GradLoss: {:.6f} '.format(
                        epoch, batch_idx, len(dataset.data_loader_S), 100. * batch_idx / len(dataset.data_loader_S),
                        loss1.item(), loss2.item(), loss_disc.item(), entropy_loss.item(), c_loss, gmn_loss)) # 

            end_times = time.time()
            # print("One step, ", end_times-start_times)
            
        # test
        test_loss, test_acc, avg, cls_avg_acc = test(epoch, model, dataset_test, dset_classes, classes_acc, test_loader, device)


        if test_acc > best_acc:
            torch.save(model.state_dict(), opt.WEIGHTS + "BEST_{}.pth".format(epoch))
            best_acc = test_acc

        print("best acc:", best_acc)


        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("==========" * 8 + "%s\n" % (end_time))

        # save
        times = (datetime.datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(now_time,'%Y-%m-%d %H:%M:%S')).seconds
        
        info = (epoch,test_loss,test_acc,avg,best_acc,cls_avg_acc,pre_acc, now_acc,times,loss1.item())
        dfhistory.loc[epoch - 1] = info



    print("Finished Training...")
    dfhistory.to_csv(opt.LOGFILE)
    print("LOG FILE HAVE SAVED!")


def test(epoch, model, dataset_test, dset_classes, classes_acc,test_loader, device):
    """, return float,float,obj,float"""
    model.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    for batch_idx, data in enumerate(dataset_test):
        if dataset_test.stop_T:
            break
        img, label, index = data['T'].to(device), data['T_label'].to(device), data['T_index']
        img, label = Variable(img, volatile=True), Variable(label) 
        
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
    for i in dset_classes: 
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    cls_acc = classes_acc
    cls_avg_acc = np.average(avg)
    print('\tClass accuracy average:', cls_avg_acc) 
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0

    return test_loss, 100. * float(correct_add) / size, avg, cls_avg_acc





if __name__ == '__main__':
    # fire.Fire()
#     draw_ana()
    train()
    # case()
    # import pandas as pd
    # data = pd.read_csv("/dfs/data/DA/data/digit/list/mnist_train.txt")
    # print(data.shape)