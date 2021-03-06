from __future__ import print_function

import argparse
import socket
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list

from eval.meta_eval import meta_test
from util import print_versions
print_versions()

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    # parser.add_argument('--model_path', type=str, default='/data/jiacheng/rfs/models_distilled/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:1.0_b:0_trans_A_born1/resnet12_last.pth', help='absolute path to .pth model')
    # parser.add_argument('--model_path', type=str, default='/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_22/resnet12_last.pth')
    # parser.add_argument('--model_path', type=str, default='/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_x3/resnet12_last.pth')
    # parser.add_argument('--model_path', type=str, default='/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_4/resnet12_last.pth')
    parser.add_argument('--model_path', type=str, default='/data/jiacheng/rfs/models_distilled/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:1.0_b:0_trans_A_born1/resnet12_last.pth')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # specify data_root
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=3000, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=20, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=0, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--classifier', type=str, default='Cosine', help='type of used classifier', choices=['kNN', 'LDA','QDA', 'LR', 'NN', 'Cosine', 'SGB', 'CVGB', 'AdaBoost', 'SVM', 'bagging', 'ensemble', 'LabelSpreading', 'LabelPropagation'])
    parser.add_argument('--l2_normalize', default=False)
   
    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    return opt


import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

if __name__ == '__main__':
    opt = parse_option()
    opt.n_gpu = torch.cuda.device_count()

    # test loader
    args = opt
    pprint(vars(args))

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_test_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    # load model
    model = create_model(opt.model, n_cls, opt.dataset)
    model.l2_normalize = opt.l2_normalize

    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt['model'])

    # import numpy as np
    # np.save('classifier_weight.npy', model.classifier.weight.cpu().detach().numpy())



    if torch.cuda.is_available():
        model = model.cuda()
        if opt.n_gpu>1:
            model = torch.nn.DataParallel(model)


        cudnn.benchmark = True


    # import warnings
    # warnings.filterwarnings('ignore') 

    # model_list = [
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_3/resnet12_last.pth',
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_4/resnet12_last.pth',
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_5/resnet12_last.pth']

    # model_list = [
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_5/resnet12_last.pth']
    # model_list = [
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_x1/resnet12_last.pth']


    # model_list = ['/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_3/resnet12_last.pth']
    # model_list = ['/data/jiacheng/rfs/models_distilled/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:1.0_b:0_trans_A_born1/resnet12_last.pth']
    model_list = None

    # model_list = [
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_21/resnet12_last.pth',
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_22/resnet12_last.pth',
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_23/resnet12_last.pth',
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_24/resnet12_last.pth']

    # model_list = [
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_x2/resnet12_last.pth',
    # '/data/jiacheng/rfs/models_pretrained/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_x3/resnet12_last.pth']



    # evalation
    # start = time.time()
    # val_acc, val_std = meta_test(model, meta_valloader, classifier=opt.classifier)
    # val_time = time.time() - start
    # print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std, val_time))

    # start = time.time()
    # val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=False, classifier=opt.classifier)
    # val_time = time.time() - start
    # print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat, val_std_feat, val_time))

    # start = time.time()
    # test_acc, test_std = meta_test(model, meta_testloader, classifier=opt.classifier, model_list=model_list)
    # test_time = time.time() - start
    # print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

    # start = time.time()
    # test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False, classifier=opt.classifier, model_list=model_list)
    # test_time = time.time() - start
    # print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))


    # start = time.time()
    # test_acc, test_std = meta_test(model, meta_testloader, is_norm=False, classifier=opt.classifier, model_list=model_list)
    # test_time = time.time() - start
    # print('test_acc (no normalization): {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False, is_norm=False, classifier=opt.classifier)
    test_time = time.time() - start
    print('test_acc_feat (no normalization): {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))