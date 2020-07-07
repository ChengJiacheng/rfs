from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.nn.functional as F

from util import normalize
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sgb import  DeepBoosting4, DeepXGBoosting, DeepXGBoosting_predict, DeepXGBoosting_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h



def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR', model_list=None):
    net = net.eval()
    # net.train()
    acc = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:

                if model_list:
                    temp = 1

                    for model_path in model_list:
                        ckpt = torch.load(model_path)
                        net.load_state_dict(ckpt['model'])

                        if temp:
                            temp = 0
                            support_features = net(support_xs).view(support_xs.size(0), -1)
                            query_features = net(query_xs).view(query_xs.size(0), -1)
                            # output = torch.nn.functional.softmax(model(input), dim=1)
                            
                            # support_features = normalize(support_features)
                            # query_features = normalize(query_features)

                        else:
                            # print(2)

                            
                            support_features += net(support_xs).view(support_xs.size(0), -1)
                            query_features += net(query_xs).view(query_xs.size(0), -1)        

                            # support_features = torch.cat((support_features, net(support_xs).view(support_xs.size(0), -1)), dim=0)

                            # support_features += normalize(net(support_xs).view(support_xs.size(0), -1))
                            # query_features += normalize(net(query_xs).view(query_xs.size(0), -1))   

                    # support_ys = support_ys.repeat((1, len(model_list)))       
                    query_features /= len(model_list)

                else:

                    support_features = net(support_xs).view(support_xs.size(0), -1)
                    query_features = net(query_xs).view(query_xs.size(0), -1)

                    # support_features = F.softmax(net(support_xs).view(support_xs.size(0), -1), dim=-1)
                    # query_features = F.softmax(net(query_xs).view(query_xs.size(0), -1), dim=-1)
            else:


                if model_list:
                    temp = 1

                    for model_path in model_list:
                        ckpt = torch.load(model_path)
                        net.load_state_dict(ckpt['model'])

                        feat_support, _ = net(support_xs, is_feat=True)
                        feat_query, _ = net(query_xs, is_feat=True)

                        if temp:

                            support_features = feat_support[-1].view(support_xs.size(0), -1)
                            query_features = feat_query[-1].view(query_xs.size(0), -1)


                        else:

                            support_features += feat_support[-1].view(support_xs.size(0), -1)
                            query_features += feat_query[-1].view(query_xs.size(0), -1) 
                else:   
                    feat_support, _ = net(support_xs, is_feat=True)
                    feat_query, _ = net(query_xs, is_feat=True)

                    support_features = feat_support[-1].view(support_xs.size(0), -1)
                    query_features = feat_query[-1].view(query_xs.size(0), -1)

                    # for i in range(5):
                    #     feat_support, _ = net(support_xs, is_feat=True)
                    #     feat_query, _ = net(query_xs, is_feat=True)
                        
                    #     support_features += feat_support[-1].view(support_xs.size(0), -1)
                    #     query_features += feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            from scipy.spatial import distance
            pdist = distance.cdist(support_features, support_features, 'cosine')
            import matplotlib.pyplot as plt
            from scipy.io import savemat
            features = np.vstack((support_features, query_features))
            labels = np.concatenate((support_ys, query_ys))
            savemat("result.mat", {'features': features, 'labels': labels})


            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='sag', max_iter=100, multi_class='multinomial', tol=1e-4, verbose=0, fit_intercept=False, C=1, penalty='l2')
                clf.fit(support_features, support_ys)

                query_ys_pred = clf.predict(query_features)

                

                query_proba = clf.predict_proba(query_features)
                query_proba = np.max(query_proba, axis=1)
                idx_pseudo = np.where(query_proba>0.25)[0]
                print(len(idx_pseudo))

                # query_ys[idx_pseudo] == query_ys_pred[idx_pseudo]

                clf = LogisticRegression(random_state=0, solver='sag', max_iter=100, multi_class='multinomial', tol=1e-4, verbose=0, fit_intercept=False, C=1, penalty='l2')
                clf.fit(np.vstack((query_features[idx_pseudo], support_features)), np.concatenate((query_ys_pred[idx_pseudo], support_ys), axis=None))
                query_ys_pred = clf.predict(query_features)

                # clf.score(query_features, query_ys)
                # acc.append(clf.score(query_features, query_ys))

                # print("train acc: %f"%(clf.score(support_features, support_ys)) )

            elif classifier == 'gcn':
                from pygcn.models import GCN
                gma = torch.tensor(np.float32(1 * np.ones(adj.shape[0])))

                gcn = GCN(nfeat=features.shape[1],
                            nhid=16,
                            nclass=5,
                            dropout=0.5, 
                            gma = gma,
                            learnable = False,
                            normalization = True,
                            renormalization = False)

                def train(epoch, best_acc, best_model):
                    t = time.time()
                    gcn.train()
                    output = gcn(features, adj)
                    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                    acc_train = accuracy(output[idx_train], labels[idx_train])
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                
                
                    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                    acc_val = accuracy(output[idx_val], labels[idx_val])
                    print('Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))

                        
                    return 0, model.state_dict()
                        
                
                
                def test():
                    gcn.eval()
                    output = gcn(features, adj)
                    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
                    acc_test = accuracy(output[idx_test], labels[idx_test])
                    print("Test set results:",
                        "loss= {:.4f}".format(loss_test.item()),
                        "accuracy= {:.4f}".format(acc_test.item()))
                    return acc_test.item()
                    
                optimizer = optim.Adam([
                                {'params': gcn.parameters()}
                #                {'params': gcn.gma, 'lr': 1e1}
                            ], lr=0.01, weight_decay=5e-4)

                # Train model
                t_total = time.time()
                best_acc = 0
                best_model = model.state_dict()
                #1/0
                for epoch in range(100):
                    best_acc, best_model = train(epoch, best_acc, best_model)
                print("Optimization Finished!")
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
                gcn.load_state_dict(best_model)


            elif classifier == 'LabelSpreading':
                from sklearn.semi_supervised import LabelSpreading
                # label_prop_model = LabelSpreading(kernel = 'rbf', gamma= 50)
                label_prop_model = LabelSpreading(kernel = 'knn', n_neighbors = 3)
                # label_prop_model.fit(support_features, support_ys)
                label_prop_model.fit(np.vstack((support_features,query_features)), np.vstack((support_ys.reshape(-1,1), np.zeros((query_features.shape[0], 1))-1)).ravel() )

                query_ys_pred = label_prop_model.predict(query_features)
                acc.append(label_prop_model.score(query_features, query_ys))

            elif classifier == 'LabelPropagation':
                from sklearn.semi_supervised import LabelPropagation
                label_prop_model = LabelPropagation()
                # label_prop_model.fit(support_features, support_ys)
                label_prop_model.fit(np.vstack((support_features,query_features)), np.vstack((support_ys.reshape(-1,1), np.zeros((query_features.shape[0], 1))-1)).ravel() )

                query_ys_pred = label_prop_model.predict(query_features)
                acc.append(label_prop_model.score(query_features, query_ys))
                
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'kNN':
                clf = KNeighborsClassifier(n_neighbors=1)

                clf.fit(support_features, support_ys)
                query_ys_pred = NN(support_features, support_ys, query_features)
                acc.append(clf.score(query_features, query_ys))

            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)

                # query_proba = clf.predict_proba(query_features)
                # query_proba = np.max(query_proba, axis=1)
                # idx_pseudo = np.where(query_proba>0.4)[0]
                # print(len(idx_pseudo))

                # # query_ys[idx_pseudo] == query_ys_pred[idx_pseudo]

                # clf = LogisticRegression(random_state=0, solver='sag', max_iter=100, multi_class='multinomial', tol=1e-4, verbose=0, fit_intercept=False, C=1, penalty='l2')
                # clf.fit(np.vstack((query_features[idx_pseudo], support_features)), np.concatenate((query_ys_pred[idx_pseudo], support_ys), axis=None))
                # query_ys_pred = clf.predict(query_features)

                # # clf.score(query_features, query_ys)
                # # acc.append(clf.score(query_features, query_ys))

                # # print("train acc: %f"%(clf.score(support_features, support_ys)) )

            elif classifier == 'SGB':
                gbes_grow=DeepBoosting4(support_features, support_ys, TrainMethod="GrowDeep", n_estimators=5000000,GrowDeep_max_iterPerDepthNUM=500, GrowDeep_max_depthNUM=50, GrowDeep_max_no_improvement=3, 
                            GrowDeep_tol_no_improvement=0.00001,GrowDeep_init_depth=1,AllowGrowDeepRetrain=1, validation_fraction=0.2,n_iter_no_change=100,tol=0.01,tolAdjust=1,random_state=0,
                            FixedDepth_max_depth=50, learning_rate=0.01,verbose=1, CrossVali_random_state=1, CrossVali_n_splits=3,
                            CrossVali_max_depth_list=[1], CrossVali_verbose=2)

                # print(gbes_grow.score(support_features, support_ys))
                acc.append(gbes_grow.score(query_features, query_ys))
            elif classifier == 'CVGB':
                grid_searchNumiter=DeepBoosting4(support_features, support_ys, TrainMethod="CrossValidateDepthAndNumIterations", n_estimators=5000000,GrowDeep_max_iterPerDepthNUM=500, GrowDeep_max_depthNUM=50, GrowDeep_max_no_improvement=3, 
                    GrowDeep_tol_no_improvement=0.00001,GrowDeep_init_depth=1,AllowGrowDeepRetrain=1, validation_fraction=0.2,n_iter_no_change=20,tol=0.01,random_state=0,
                    FixedDepth_max_depth=50, learning_rate=0.01,verbose=0, CrossVali_random_state=1, CrossVali_n_splits=3,
                    CrossVali_max_depth_list=[1,2,3,4,5,6,7,8,9,10,20,50], CrossVali_n_estimators_list=[100,250,500,750,1000,1500,2000], CrossVali_verbose=0)
                acc.append(grid_searchNumiter.best_estimator_.score(query_features, query_ys))
            elif classifier == 'AdaBoost':
                # base_estimator = AdaBoostClassifier(n_estimators=20, learning_rate=.1, random_state=0)
                # base_estimator = LogisticRegression(solver='sag', max_iter=100, multi_class='multinomial', verbose=0) 
                # clf = AdaBoostClassifier(base_estimator = base_estimator, n_estimators=1, learning_rate=.1, random_state=0)

                clf = AdaBoostClassifier(n_estimators=20, learning_rate=.1, random_state=0)


                clf.fit(support_features, support_ys)
                acc.append(clf.score(query_features, query_ys))

                # print("train acc: %f"%(clf.score(support_features, support_ys)) )
            elif classifier == 'LDA':
                clf = LinearDiscriminantAnalysis(shrinkage='auto', n_components = 8)     
                clf.fit(support_features, support_ys)          
                acc.append(clf.score(query_features, query_ys))                 

            elif classifier == 'QDA':
                clf = QuadraticDiscriminantAnalysis(reg_param=1)     
                clf.fit(support_features, support_ys)          
                acc.append(clf.score(query_features, query_ys))               

            elif classifier == 'SVM':
                clf = SVC(C=1, kernel='rbf', gamma='scale', decision_function_shape = 'ovr', probability=True)

                # clf = SVC(C=1, gamma='scale', decision_function_shape = 'ovr')
                clf.fit(support_features, support_ys)
                acc.append(clf.score(query_features, query_ys))               
            elif classifier == 'bagging':
                # clf = BaggingClassifier(base_estimator=SVC(), 
                # max_features=64, n_jobs=2, max_samples=0.9, n_estimators=1, random_state=0).fit(support_features, support_ys)
                clf = BaggingClassifier(base_estimator=LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial'), 
                max_features=1.0, n_jobs=1, max_samples=1.0, n_estimators=1, random_state=0, verbose=0).fit(support_features, support_ys)
                        
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'ensemble':
                base_clf = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial')
                # base_clf = SVC(C=1, gamma='scale', decision_function_shape = 'ovr', probability=True)
                # base_clf = SVC(C=1, kernel='linear', gamma='scale', decision_function_shape = 'ovr', probability=True)

                for i in range(50):
                    weak_clf = LogisticRegression(random_state=i, solver='lbfgs', max_iter=100, multi_class='multinomial')
                    weak_clf.fit(support_features, support_ys)

                    if i == 0:
                        query_prob = weak_clf.predict_proba(query_features)
                    else:
                        query_prob += weak_clf.predict_proba(query_features)


                query_ys_pred = np.argmax(query_prob, axis=1)  
                
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))


            
            if classifier in ['LR', 'NN', 'Cosine', 'bagging', 'ensemble']:
                acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            if idx%10 == 0:
                print(mean_confidence_interval(acc))
                

    return mean_confidence_interval(acc)


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
