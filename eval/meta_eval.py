from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sgb import  DeepBoosting4, DeepXGBoosting, DeepXGBoosting_predict, DeepXGBoosting_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    net = net.eval()
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
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)
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
                # clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
                base_estimator = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial') 
                clf = AdaBoostClassifier(base_estimator = base_estimator, n_estimators=10, learning_rate=.1, random_state=0)
                
                clf.fit(support_features, support_ys)
                acc.append(clf.score(query_features, query_ys))
            elif classifier == 'SVM':
                clf = SVC(C=1, kernel='linear', gamma='scale', decision_function_shape = 'ovr', probability=True)

                # clf = SVC(C=1, gamma='scale', decision_function_shape = 'ovr')
                clf.fit(support_features, support_ys)
                acc.append(clf.score(query_features, query_ys))               
            elif classifier == 'bagging':
                # clf = BaggingClassifier(base_estimator=SVC(), 
                # max_features=64, n_jobs=2, max_samples=0.9, n_estimators=1, random_state=0).fit(support_features, support_ys)
                clf = BaggingClassifier(base_estimator=LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial'), 
                max_features=1.0, n_jobs=2, max_samples=1.0, n_estimators=50, random_state=0, verbose=0).fit(support_features, support_ys)
                        
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'ensemble':
                base_clf = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial')
                # base_clf = SVC(C=1, gamma='scale', decision_function_shape = 'ovr', probability=True)
                # base_clf = SVC(C=1, kernel='linear', gamma='scale', decision_function_shape = 'ovr', probability=True)

                clf1 = base_clf
                clf1.fit(support_features, support_ys)
                query_prob = clf1.predict_proba(query_features)

                # clf2 = base_clf
                # clf2.fit(support_features, support_ys)
                # query_prob += clf2.predict_proba(query_features)         

                # clf3 = base_clf
                # clf3.fit(support_features, support_ys)
                # query_prob += clf3.predict_proba(query_features)          

                # clf4 = base_clf
                # clf4.fit(support_features, support_ys)
                # query_prob += clf4.predict_proba(query_features)      

                # clf5 = base_clf
                # clf5.fit(support_features, support_ys)
                # query_prob += clf5.predict_proba(query_features)      
                # query_prob = query_prob/ 2            
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
