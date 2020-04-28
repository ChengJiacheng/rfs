# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:47:03 2020

@author: hmasnadishirazi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:14:11 2020

@author: hmasnadishirazi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 08:09:13 2020

@author: hmasnadishirazi
"""


import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import  make_scorer, accuracy_score, confusion_matrix, log_loss
from xgboost import XGBClassifier
import xgboost
import time

n_jobs = 32
"""
########## 
DEEP BOOSTING
# This version "4" of deep boosting has early stopping that we implement ourselves but the difference with version 3 is that 
the ""tol" param is decreased (adjusted) when the "NEWSCOREdepth" is less that 0.99 since an improvement of tol=0.01 is not possible anymore so 
the "tol" should be decreased to for example 0.001. Also, if "NEWSCOREdepth"<0.999 then "tol should be decreased to for example 0.0001 and so on....
tolAdjust=1 enables this adaptive "tol" setting


X_train: train data array (rows=samples, cols=features)
y_train: sample true labels
TrainMethod: a string that tells code which method of training to use:
            "GrowDeep"= Staged Gradient boosting method (Our Novel Method!)
            "CrossValidateDepth"=  gradient boosting with cross validate only tree depth Number of boosting iterations is found automatically with early stopping
            "FixedDepth" = gradient boost without cross validation on tree depth. Number of boosting iterations is found automatically with early stopping
            "CrossValidateDepthAndNumIterations"= gradient boost with cross validate both tree depth and number of iterations
n_estimators: maximum number of allowed boosting iterations. Is ignored if TrainMethod=="GrowDeep"
GrowDeep_max_iterPerDepthNUM: max number of allowed boosting iterations at each depth. Used only if TrainMethod=="GrowDeep"
GrowDeep_max_depthNUM: max allowed boosting tree depth. Used only if TrainMethod=="GrowDeep"     
GrowDeep_max_no_improvement: Staged Boosting will stop if there is no improvement in performance after having increased the tree depth this many times. Same as "M" in paper. Used only if TrainMethod=="GrowDeep"        
GrowDeep_tol_no_improvement: performance must improve by at least this much or else the GrowDeep_max_no_improvement will be increased. Used only if TrainMethod=="GrowDeep"    
GrowDeep_init_depth: initial tree depth to start with. Usually set to 1. Used only if TrainMethod=="GrowDeep"  
AllowGrowDeepRetrain: Ignore for now.
validation_fraction: fraction of data used for automatic early stopping number of boosting iterations. Used only if TrainMethod=="FixedDepth" or "CrossValidateDepth"
n_iter_no_change: automatically stop the number of boosting iterations if there is no improvement in perforamnce for this many iterations. Used only if TrainMethod=="GrowDeep" , "FixedDepth" or "CrossValidateDepth"
tol: The cumulative sum of performance improvements must be initially grerater than this or the early stopping algorithm with stop boosting iterations.   Used only if TrainMethod=="GrowDeep" , "FixedDepth" or "CrossValidateDepth"
tolAdjust=if 1 enables adaptive "tol" setting.
LossEarlyStop: Loss function to compute the early stopping performance. can be "accuracy" or "logloss". Works much better with "logloss". 
random_state: random state for validation data spliting.
FixedDepth_max_depth: max tree depth used only in TrainMethod=="FixedDepth"
learning_rate: learning rate of gradient boosting used by all methods. 
verbose: print performance at each train iteration for Staged Boosting method. used in TrainMethod=="GrowDeep"
CrossVali_random_state: ransom state for cross validation methods. Used only in TrainMethod=="CrossValidateDepthAndNumIterations" or "CrossValidateDepth"
CrossVali_n_splits: number of cross validation folds.Used only in TrainMethod=="CrossValidateDepthAndNumIterations" or "CrossValidateDepth"
CrossVali_max_depth_list: list of tree depths to cross validate over. Used only in TrainMethod=="CrossValidateDepthAndNumIterations" or "CrossValidateDepth"
CrossVali_n_estimators_list: list of boosting iterations to cross validate over. Used only in TrainMethod=="CrossValidateDepthAndNumIterations" or "CrossValidateDepth"
CrossVali_verbose: print perforamnce at each cross validation for cross validation methods. TrainMethod=="CrossValidateDepthAndNumIterations" or "CrossValidateDepth"
"""
def DeepBoosting4(X_train, y_train, TrainMethod="GrowDeep", n_estimators=5000000, GrowDeep_max_iterPerDepthNUM=500,  GrowDeep_max_depthNUM=50, GrowDeep_max_no_improvement=3, 
            GrowDeep_tol_no_improvement=0.00001,GrowDeep_init_depth=1, AllowGrowDeepRetrain=1, validation_fraction=0.2,n_iter_no_change=5,tol=0.01,tolAdjust=1,LossEarlyStop="logloss",random_state=0,
            FixedDepth_max_depth=50, learning_rate=0.01,verbose=0, CrossVali_random_state=1, CrossVali_n_splits=2,
            CrossVali_max_depth_list=[1], CrossVali_n_estimators_list=[100,250,500,750,1000], CrossVali_verbose=2):
    
    
    if TrainMethod=="CrossValidateDepth":
        gbes_shallow = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,validation_fraction=validation_fraction,
                                n_iter_no_change=n_iter_no_change,tol=tol,random_state=random_state,learning_rate=learning_rate)
    
        param_grid = {'max_depth': CrossVali_max_depth_list} #[1,2,3,4,5,6,7,8,9,10]
        scorers = {'accuracy_score': make_scorer(accuracy_score)}
        refit_score='accuracy_score'
        skf = StratifiedKFold(n_splits=CrossVali_n_splits, random_state=CrossVali_random_state)
        grid_searchshallow = GridSearchCV(gbes_shallow, param_grid, scoring=scorers, refit=refit_score,cv=skf, 
                                          return_train_score=True, n_jobs=n_jobs, verbose=CrossVali_verbose)
        grid_searchshallow.fit(X_train, y_train)
        if verbose==1:
            print("print(grid_searchshallow.best_params_)=",grid_searchshallow.best_params_)
            print("grid_searchv.score(X_train, y_train)=", grid_searchshallow.best_estimator_.score(X_train, y_train))
            print("n_estimators =", grid_searchshallow.best_estimator_.n_estimators_)
        return grid_searchshallow
        
    if TrainMethod=="FixedDepth":
        gbes_deeptree = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, max_depth=FixedDepth_max_depth,
                                                 validation_fraction=validation_fraction,n_iter_no_change=n_iter_no_change,tol=tol,
                                                 random_state=random_state,learning_rate=learning_rate)
        gbes_deeptree.fit(X_train, y_train)
        if verbose==1:
            print("gbes_deeptree.score(X_train, y_train)=", gbes_deeptree.score(X_train, y_train))
            print("n_estimators =", gbes_deeptree.n_estimators_) 
        return gbes_deeptree
    
    if TrainMethod=="CrossValidateDepthAndNumIterations":
        gbes_shallowNumIters = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,random_state=random_state,learning_rate=learning_rate)
    
        param_grid = {'max_depth': CrossVali_max_depth_list, 'n_estimators': CrossVali_n_estimators_list} #[1,2,3,4,5,6,7,8,9,10]
        scorers = {'accuracy_score': make_scorer(accuracy_score)}
        refit_score='accuracy_score'
        skf = StratifiedKFold(n_splits=CrossVali_n_splits, random_state=CrossVali_random_state)
        gbes_searchshallowNumIters = GridSearchCV(gbes_shallowNumIters, param_grid, scoring=scorers, refit=refit_score,cv=skf, 
                                          return_train_score=True, n_jobs=n_jobs, verbose=CrossVali_verbose)
        gbes_searchshallowNumIters.fit(X_train, y_train)
        if verbose==1:
            print("print(gbes_searchshallowNumIters.best_params_)=",gbes_searchshallowNumIters.best_params_)
            print("grid_searchvNumIters.score(X_train, y_train)=", gbes_searchshallowNumIters.best_estimator_.score(X_train, y_train))
            print("n_estimators =", gbes_searchshallowNumIters.best_estimator_.n_estimators_)
        return gbes_searchshallowNumIters
            
        
    if TrainMethod=="GrowDeep":
        TotalNumWeakLearners=0
        gbes_grow = ensemble.GradientBoostingClassifier(n_estimators=1, max_depth=GrowDeep_init_depth, random_state=random_state, 
                    warm_start=True, learning_rate=learning_rate)
        gbes_grow.fit(X_train, y_train)
        if LossEarlyStop=="accuracy":
            NEWSCORE=gbes_grow.score(X_train, y_train)
        if LossEarlyStop=="logloss":
            y_pred=gbes_grow.predict_proba(X_train)
            NEWSCORE=1-log_loss(y_train, y_pred) 
        TotalNumWeakLearners+=1
        if TotalNumWeakLearners>=n_estimators or NEWSCORE==1.0:
            print("NEWSCORE=100%")
            return gbes_grow
        ##fit with early stop##
        no_improvement_counter_EarlyStop=0
        DIFFSCORERUNSUM=0
        for iterNUM in range(GrowDeep_max_iterPerDepthNUM):
            #_ = gbes_grow.set_params(n_estimators=1,  warm_start=True)  # set warm_start and new params of trees
            gbes_grow.n_estimators += 1
            _ = gbes_grow.fit(X_train, y_train) # fit additional  trees to est
            TotalNumWeakLearners+=1
            if TotalNumWeakLearners>=n_estimators or NEWSCORE==1.0:
                print("NEWSCORE=100%")
                return gbes_grow
            OLDSCORE=NEWSCORE
            if LossEarlyStop=="accuracy":
                NEWSCORE=gbes_grow.score(X_train, y_train)
            if LossEarlyStop=="logloss":
                y_pred=gbes_grow.predict_proba(X_train)
                NEWSCORE=1-log_loss(y_train, y_pred) 
            DIFFSCORE=NEWSCORE-OLDSCORE
            DIFFSCORERUNSUM+=DIFFSCORE
            if verbose>=2:
                print("NEWSCORE at each early stop=",NEWSCORE)
                print("DIFFSCORE at each early stop=",DIFFSCORE)
                print("DIFFSCORERUNSUM at each early stop=",DIFFSCORERUNSUM)
            if (DIFFSCORERUNSUM)>tol:
                no_improvement_counter_EarlyStop=0 # reset this counter if there is improvement.
                DIFFSCORERUNSUM=0
            if (DIFFSCORERUNSUM)<tol:
                no_improvement_counter_EarlyStop+=1
            if no_improvement_counter_EarlyStop==n_iter_no_change:
                break
        if verbose>=1:
            print("n_estimators for depth"+str(1)+"=", gbes_grow.n_estimators_)
            print("TotalNumWeakLearners", TotalNumWeakLearners)
            print("NEWSCORE", NEWSCORE)
        ##fit with early stop##
        #if LossDepth=="accuracy":
        #    NEWSCOREdepth=gbes_grow.score(X_train, y_train)
        #if LossDepth=="logloss":
        #    y_pred=gbes_grow.predict_proba(X_train)
        #    NEWSCOREdepth=1-log_loss(y_train, y_pred) 
        NEWSCOREdepth= gbes_grow.score(X_train, y_train) #NEWSCORE
        if tolAdjust==1 and (1-NEWSCOREdepth)<tol:
            tol=tol/2   
        if verbose>=1:
            print("NEWSCOREdepth",NEWSCOREdepth)
        if NEWSCOREdepth==1.0:
                return gbes_grow
        no_improvement_counter=0
        RetrainFLAG=0
        for depthNUM in range(GrowDeep_max_depthNUM):
            _ = gbes_grow.set_params(max_depth=depthNUM+2,  warm_start=True)  # set warm_start and new params of trees
            ##fit with early stop##
            no_improvement_counter_EarlyStop=0
            DIFFSCORERUNSUM=0
            for iterNUM in range(GrowDeep_max_iterPerDepthNUM):
                #_ = gbes_grow.set_params(n_estimators=1,  warm_start=True)  # set warm_start and new params of trees
                gbes_grow.n_estimators += 1
                _ = gbes_grow.fit(X_train, y_train) # fit additional  trees to est
                TotalNumWeakLearners+=1
                if TotalNumWeakLearners>=n_estimators or NEWSCORE==1.0:
                    print("NEWSCORE=100%")
                    return gbes_grow
                OLDSCORE=NEWSCORE
                if LossEarlyStop=="accuracy":
                    NEWSCORE=gbes_grow.score(X_train, y_train)
                if LossEarlyStop=="logloss":
                    y_pred=gbes_grow.predict_proba(X_train)
                    NEWSCORE=1-log_loss(y_train, y_pred) 
                DIFFSCORE=NEWSCORE-OLDSCORE
                DIFFSCORERUNSUM+=DIFFSCORE
                if (DIFFSCORERUNSUM)>tol:
                    no_improvement_counter_EarlyStop=0 # reset this counter if there is improvement.
                    DIFFSCORERUNSUM=0
                if (DIFFSCORERUNSUM)<tol:
                    no_improvement_counter_EarlyStop+=1
                if no_improvement_counter_EarlyStop==n_iter_no_change:
                    break
            if verbose>=1:
                print("n_estimators for depth"+str(depthNUM+2)+"=", gbes_grow.n_estimators_)
                print("TotalNumWeakLearners", TotalNumWeakLearners)
                print("NEWSCORE", NEWSCORE)
            ##fit with early stop##
            OLDSCOREdepth=NEWSCOREdepth
            #if LossDepth=="accuracy":
            #    NEWSCOREdepth=gbes_grow.score(X_train, y_train)
            #if LossDepth=="logloss":
            #    y_pred=gbes_grow.predict_proba(X_train)
            #    NEWSCOREdepth=1-log_loss(y_train, y_pred) 
            NEWSCOREdepth=gbes_grow.score(X_train, y_train) #NEWSCORE 
            if tolAdjust==1 and (1-NEWSCOREdepth)<tol:
                tol=tol/2  
            if verbose>=1:
                print("NEWSCOREdepth",NEWSCOREdepth)
            if NEWSCOREdepth==1.0:
                break
            DIFFSCOREdepth=NEWSCOREdepth-OLDSCOREdepth
            if DIFFSCOREdepth>=0 and (DIFFSCOREdepth)>GrowDeep_tol_no_improvement:
                no_improvement_counter=0 # reset this counter if there is improvement.
            if DIFFSCOREdepth>=0 and (DIFFSCOREdepth)<GrowDeep_tol_no_improvement:
                no_improvement_counter+=1
                if no_improvement_counter==GrowDeep_max_no_improvement:
                    break
            if DIFFSCOREdepth<0 :
                Retrain_iter=depthNUM+1
                RetrainFLAG=1
                break
            
        
        return gbes_grow






"""
########## 
DEEP XG BOOSTING

WARNING: only works with xgboost version 0.82 and higher!
Note: XGBoos'ts internal early stoping does not work with scikitlearn's gridsearch cross validation!
##########
"""
def DeepXGBoosting(X_train, y_train, TrainMethod="CrossValidateDepth", n_estimators=200, GrowDeep_max_depthNUM=50, GrowDeep_max_no_improvement=3, 
            GrowDeep_tol_no_improvement=0.00001,GrowDeep_init_depth=1, AllowGrowDeepRetrain=1,n_iter_no_change=5, random_state=0,
            FixedDepth_max_depth=50, learning_rate=0.01,verbose=0, CrossVali_random_state=1, CrossVali_n_splits=2,
            CrossVali_max_depth_list=[1], CrossVali_n_estimators_list=[100,250,500,750,1000], CrossVali_verbose=2, GrowDeep_verbose=True):
    
    
    if TrainMethod=="CrossValidateDepth":
        
        gbes_shallow = XGBClassifier( learning_rate=learning_rate,  n_estimators=n_estimators, importance_type="gain")  
        param_grid = {'max_depth': CrossVali_max_depth_list, 'n_estimators': CrossVali_n_estimators_list} #[1,2,3,4,5,6,7,8,9,10]
        scorers = {'accuracy_score': make_scorer(accuracy_score)}
        refit_score='accuracy_score'
        skf = StratifiedKFold(n_splits=CrossVali_n_splits, random_state=CrossVali_random_state)
        grid_searchshallow = GridSearchCV(gbes_shallow, param_grid, scoring=scorers, refit=refit_score,cv=skf, 
                                          return_train_score=True, n_jobs=n_jobs, verbose=CrossVali_verbose)
        grid_searchshallow.fit(X_train, y_train)
        if verbose==1:
            print("print(grid_searchshallow.best_params_)=",grid_searchshallow.best_params_)
            print("grid_searchv.score(X_train, y_train)=", grid_searchshallow.best_estimator_.score(X_train, y_train))
            print("n_estimators =", grid_searchshallow.best_estimator_.n_estimators)
        return grid_searchshallow
        
    if TrainMethod=="FixedDepth":
        gbes_deeptree = XGBClassifier( learning_rate=learning_rate,  n_estimators=n_estimators, max_depth=FixedDepth_max_depth, importance_type="gain")  
        gbes_deeptree.fit(X_train, y_train)
        if verbose==1:
            print("gbes_deeptree.score(X_train, y_train)=", gbes_deeptree.score(X_train, y_train))
            print("n_estimators =", gbes_deeptree.n_estimators) 
        return gbes_deeptree

    
 
"""
Do prediction for "GrowDeep" XGBoosting model trained with "GrowDeep"
NOTE: Must use this for predicting on a test set because if you use the default predict() you will not get correct results since it will use only the weal earners up to the first early stopping and NOT all weak learners.
"""    
def DeepXGBoosting_predict(gbes_grow, X_test):
    y_pred=gbes_grow.predict(X_test, ntree_limit=0) # NOTE: must set ntree_limit=0 since if you dont it will default to using only the number of weak learners after first early stopping rather than all weak learners. 
    return y_pred

"""
Find Accuracy on test set for "GrowDeep" XGBoosting model trained with "GrowDeep"
NOTE: Must use this for accuracy on a test set because if you use the default predict() you will not get correct results since it will use only the weal earners up to the first early stopping and NOT all weak learners.
"""    
def DeepXGBoosting_score(gbes_grow, X_test, y_test):
    y_pred=gbes_grow.predict(X_test, ntree_limit=0) # NOTE: must set ntree_limit=0 since if you dont it will default to using only the number of weak learners after first early stopping rather than all weak learners. 
    TTTT=confusion_matrix(y_test, y_pred)
    Accuracy=(TTTT[0,0]+TTTT[1,1])/(len(y_test))
    return Accuracy

