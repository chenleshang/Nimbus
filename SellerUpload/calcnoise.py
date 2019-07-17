#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:37:18 2019

@author: leshang
"""
#%%
import numpy as np
import logging
from datetime import datetime
from numpy import genfromtxt
import copy
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from md5 import md5
import sqlite3
from werkzeug.utils import secure_filename
import os
import pandas as pd  
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sklearn
from copy import deepcopy

def get_mse_score(reg, TestX, TestY):
    predY = reg.predict(TestX)
    return mean_squared_error(TestY, predY)

def PRO_CanModel(PriceUp,Price,rho):
#    q = Price;
#    PRO = np.sum( np.multiply(np.multiply(rho, q), (q - 1e-10 <= PriceUp) ) ) ;
    PRO = np.sum( np.multiply(np.multiply(rho, Price), (Price - 1e-10 <= PriceUp) ) ) ;
    return PRO

def MBP_CanModel_DP(Delta, PriceUp, rho):
    Delta= Delta * 1.0
    delta = Delta
    K = np.shape(PriceUp)[0]
    PROSet = np.zeros((K, 1))
    PROSetFix = np.zeros((K, 1))
    PROPriceSet = np.matrix(np.zeros((K, K)))
    PROPriceSetFix = np.matrix(np.zeros((K, K)))
    PROSet[K - 1, 0] = rho[K - 1] * PriceUp[K - 1]
    PROSetFix[K - 1, 0] = rho[K - 1] * PriceUp[K - 1]
    PROPriceSet[K - 1, K - 1] = PriceUp[K - 1]
    PROPriceSetFix[K - 1, K - 1] = PriceUp[K - 1]
    p = PriceUp * 1.0
    
#    for i in range(K - 1, 0, -1):
    for i in range(K - 2, -1, -1):
#        if( p[i] / delta[i] > PROPriceSet[i + 1, i + 1] / delta[i+1] ):
#            PROPriceSet[i, :] = PROPriceSet[i + 1 , :]
#            PROPriceSet[i, i] = p[i]
#            PROPriceSetFix[i, :] = PROPriceSet[i, :]
#            PROSet[i] = PROSet[i + 1 ] + rho[i] * p[i]
#            PROSetFix[i] = PROSet[i]
        if( p[i] / delta[i] > PROPriceSet[i + 1, i + 1] / delta[i+1] ):
            PROPriceSet[i, :] = deepcopy(PROPriceSet[i + 1 , :])
            PROPriceSet[i, i] = deepcopy(p[i])
            PROPriceSetFix[i, :] = deepcopy(PROPriceSet[i, :])
            PROSet[i] = PROSet[i + 1 ] + rho[i] * p[i]
            PROSetFix[i] = PROSet[i]    
        else:
            PROPriceSet[i, :] = deepcopy(PROPriceSet[i + 1, :])
            PROPriceSet[i, i] = delta[i] / delta[i + 1] * PROPriceSet[i + 1, i + 1]
            PROSet[i] = PRO_CanModel( PriceUp[i:K+1], PROPriceSet[i, i:K+1].transpose(), rho[i:K+1] )
            
            tempOptPriceSet = deepcopy(PROPriceSet[i, :])
            tempOptPriceSet[0, i] = deepcopy(p[i])
            
            fbp_temp = (p[i]/delta[i])[0,0]
            FixBestPrice = fbp_temp * delta
            
            FixBestPRO = PRO_CanModel( PriceUp[i:K+1], FixBestPrice[i:K+1], rho[i:K+1])
            
            for j in range(i+1, K, 1):
                if( p[j] < delta[j]/delta[i] * p[i] ):
#                    tempj = j
                    tempPRO = PRO_CanModel( PriceUp[i : j], tempOptPriceSet[0, i: j].transpose(), rho[i: j])+ PROSetFix[j, 0] 
#                    tempPRO = PRO_CanModel( PriceUp[i : j - 1], tempOptPriceSet[0, i: j - 1].transpose(), rho[i: j - 1] + PROSetFix[j] )
                    if(tempPRO > FixBestPRO):
                        FixBestPRO = tempPRO
                        FixBestPrice[j: K + 1] = np.matrix(PROPriceSetFix[j, j : K + 1]).T
                        FixBestPrice[i : j, 0] = np.matrix(tempOptPriceSet[0, i: j]).T
                        
                tempOptPriceSet[0, j] = delta[j, 0]/delta[i, 0] * p[i, 0]
                
            PROPriceSetFix[i, :] = np.matrix(FixBestPrice).T
            PROSetFix[i] = deepcopy(FixBestPRO)
            
            if(PROSet[i] < FixBestPRO):
                PROSet[i] = deepcopy(FixBestPRO)
                PROPriceSet[i, :] = np.matrix(FixBestPrice).T
                
    PRO = PROSet[0]
    q = PROPriceSet[0, :].transpose()
            
    return q, PRO

## Let's define the name: 
# Logistic Regression to be LogisticRegression, 
# Linear Regression to be LinearRegression, 
# kernel type: default. 
##
def calc_optimal_model(model_func, model_type, kernel_type, dataset_name, 
                       trainX, trainY, TestX, TestY, DATABASE, UPLOAD_FOLDER,
                       filetype = 'bestmodel', username = ''):
    # ### Start training process
    print('start')
    if(model_type=='LinearRegression'):
        print('Linear Regression, no intercept')
        model = model_func(fit_intercept=False)
    else:
        print('Logistic Regression, ok intercept')
        model = model_func()
    model.fit(trainX, trainY)
#    model.score(TestX, TestY)
    securedname = secure_filename(
            model_type+'_'+ kernel_type+'_' +str(datetime.now())+
            md5(pickle.dumps(model) + str(round(time.time() * 1000))).hexdigest()+ '.dump')
    
    #save file
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, filetype)):
        os.makedirs(os.path.join(UPLOAD_FOLDER, filetype))
    with open(os.path.join(UPLOAD_FOLDER, filetype, securedname), 'wb') as file:
        pickle.dump(model, file)
        
    #save metadata to DB
    #run init_db() for the first time
#    DATABASE = sqlite3.connect(DATABASE_PATH)
    insertdump = '''
        INSERT INTO bestmodel
        (username, datasetname, modeltype, kerneltype, modelpath)
        VALUES (?, ?, ?, ?, ?);
        '''
    DATABASE.execute(insertdump, 
                 (username, dataset_name, model_type, kernel_type, securedname))
    DATABASE.commit()
#    DATABASE.close()
#    print( username, dataset_name, model_type, kernel_type, securedname)
    return model
    
def show_db(DATABASE_PATH= '../../testupload/database.db'):
    conn = sqlite3.connect(DATABASE_PATH)
    insertdump = '''
        select * 
        from bestmodel
        '''
    df = pd.read_sql_query(insertdump, conn)
    print('bestmodel')
    print(df)
    insertdump = '''
    select * 
    from noisymodel
    '''
    df = pd.read_sql_query(insertdump, conn)
    print('noisymodel')
    print(df)
    conn.commit()
#    conn.close()

def add_noise(classifier, TestX, TestY, variance=1.0, linregsquareloss = None):
    """
    classifier: a trained sklearn model
    noise level: the variance of gaussian noise we're going to add
    """
    linregsquareloss = None
    new_classifier = copy.deepcopy(classifier)
    model_shape = new_classifier.coef_.shape
#    if(linregsquareloss == None):
    bias_shape = new_classifier.intercept_.shape
    # note that from this doc: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html
    # we need to multiply by the std dev 
#    if(len(model_shape) == 2):
    noise_weight = np.sqrt(variance)*np.random.randn(model_shape[0], model_shape[1])
    noise_bias = np.sqrt(variance)*np.random.randn(bias_shape[0])
#    elif(len(model_shape) == 1):
#        noise_weight = np.sqrt(variance)*np.random.randn(model_shape[0], )
#        noise_bias = np.sqrt(variance)*np.random.randn()
#    else:
#        assert(False)
    
    new_model = new_classifier.coef_ + noise_weight
    new_bias = new_classifier.intercept_ + noise_bias
    
    # load param to the model:
    new_classifier.coef_ = new_model
    new_classifier.intercept_ = new_bias
    
    new_acc = new_classifier.score(TestX, TestY)
#     TODO: modified. 
    if(linregsquareloss == get_mse_score):
#        assert(isinstance( classifier, LinearRegression))
#        print('get mse score')
        new_acc = get_mse_score(new_classifier, TestX, TestY)
    return new_acc, new_classifier

def add_noise_linreg(classifier, TestX, TestY, variance=1.0, linregsquareloss=get_mse_score):
    """
    classifier: a trained sklearn model
    noise level: the variance of gaussian noise we're going to add
    """
    new_classifier = copy.deepcopy(classifier)
    model_shape = new_classifier.coef_.shape
#    if(linregsquareloss == None):
#    bias_shape = new_classifier.intercept_.shape
    # note that from this doc: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html
    # we need to multiply by the std dev 
#    if(len(model_shape) == 2):
#        noise_weight = np.sqrt(variance)*np.random.randn(model_shape[0], model_shape[1])
#        noise_bias = np.sqrt(variance)*np.random.randn(bias_shape[0])
#    elif(len(model_shape) == 1):
    noise_weight = np.sqrt(variance)*np.random.randn(model_shape[0], )
#    noise_bias = np.sqrt(variance)*np.random.randn()
#    else:
#        assert(False)
    
    new_model = new_classifier.coef_ + noise_weight
    new_bias = new_classifier.intercept_
    
    # load param to the model:
    new_classifier.coef_ = new_model
    new_classifier.intercept_ = new_bias
    
    new_acc = new_classifier.score(TestX, TestY)
#     TODO: modified. 
    if(linregsquareloss == get_mse_score):
#        assert(isinstance( classifier, LinearRegression))
#        print('get mse score')
        new_acc = get_mse_score(new_classifier, TestX, TestY)
    return new_acc, new_classifier

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#def save_model_to_db(username, dataset_name, model_type, kernel_type, securedname, noise, error, 
#                     DATABASE_PATH):
#    #save metadata to DB
#    #run init_db() for the first time
#    conn = sqlite3.connect(DATABASE_PATH)
#    insertdump = '''
#        INSERT INTO noisymodel
#        (username, datasetname, modeltype, kerneltype, modelpath, noise, error)
#        VALUES (?, ?, ?, ?, ?);
#        '''
#    conn.execute(insertdump, 
#                 (username, dataset_name, model_type, kernel_type, securedname, noise, error))
#    conn.commit()
#    conn.close()
#    
    
def calc_error_noise_curve(model_func, model_type, kernel_type, dataset_name, 
                       trainX, trainY, TestX, TestY, optimal_model, variances, DATABASE, UPLOAD_FOLDER, progress_dict={'progress':0, 'stepprogress':0}):
#    variances = np.arange(0, 101)/100.0
    model_to_save = []
    acc_trials = []
    classifier_collector = []
    acc_plot_collector = []
    num_trials = 10
    filetype = 'noisymodel'
    #for i in range(num_trials):
    #    acc_trials.append([add_noise(clean_model, TestX, TestY, var) for var in variances])
    count = 0
    for var in variances:
        if(count % 10 == 0):
            print('calculating noisy model accuracy', count)
            progress_dict['progress'] = int(70.0 * count / len(variances) ) + progress_dict['stepprogress']
        count += 1
        
        for i in range(num_trials):
            if(model_type == 'LinearRegression'):
#                print('use add_noise_linreg')
                new_acc, new_cls = add_noise_linreg(optimal_model, TestX, TestY, var)
            else:
                new_acc, new_cls = add_noise(optimal_model, TestX, TestY, var)
            acc_trials.append(new_acc)
            classifier_collector.append(new_cls)
            
        acc_mean = np.mean(acc_trials)
        index = find_nearest_index(acc_trials, acc_mean)
        model_to_save.append(classifier_collector[index])
        #print(classifier_collector[index].score(TestX, TestY), acc_mean)
        if(model_type == 'LogisticRegression' or isinstance(model_func, sklearn.linear_model.LogisticRegression)):
            acc_plot_collector.append(classifier_collector[index].score(TestX, TestY))
        elif(model_type == 'LinearRegression' or isinstance(model_func, sklearn.linear_model.LinearRegression)):
            acc_plot_collector.append(get_mse_score(classifier_collector[index], TestX, TestY))
        else:
            print(model_type == 'LinearRegression', isinstance(model_func, sklearn.linear_model.LinearRegression), (model_func))
            assert(False)
    
    for i, model in enumerate(model_to_save):
#        print("Saving model trained on noise with variance : {} ...".format(variances[i]))
        
        securedname = secure_filename(
            model_type+'_'+ kernel_type+'_' + 'var_{}'.format(variances[i]) +
            md5(pickle.dumps(model) + str(round(time.time() * 1000))).hexdigest()+ '.dump')
        
#        with open(, 'wb') as file:
        
        if not os.path.exists(os.path.join(UPLOAD_FOLDER, filetype)):
            os.makedirs(os.path.join(UPLOAD_FOLDER, filetype))
            
        with open(os.path.join(UPLOAD_FOLDER, filetype, securedname), 'wb') as file:
            pickle.dump(model, file)
            
        # save_model_metadata_to_db
        insertdump = '''
        INSERT INTO noisymodel
        (username, datasetname, modeltype, kerneltype, modelpath, noise, error)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        '''
        DATABASE.execute(insertdump, 
             ('', dataset_name, model_type, kernel_type, securedname, variances[i], acc_plot_collector[i]))
        
    DATABASE.commit()
    if(model_type == 'LogisticRegression' or isinstance(model_func, sklearn.linear_model.LogisticRegression)):
        df = pd.DataFrame( {'variance':variances, 'accuracy': acc_plot_collector })
        df['error'] = 1 - df['accuracy']
    elif(model_type == 'LinearRegression' or isinstance(model_func, sklearn.linear_model.LinearRegression)):
        df = pd.DataFrame( {'variance':variances, 'error': acc_plot_collector })
    else:
        assert(False)
    return df[['variance', 'error']]


def find_closest(demand_error_df, error_noise_df):
    noise_collector = []
    error_collector = []
    for i, row in demand_error_df.iterrows():
        closeset_idx = find_nearest_index(error_noise_df['error'].values, row['error'])
        
        noise_collector.append(error_noise_df.iloc[closeset_idx]['variance'])
        error_collector.append(error_noise_df.iloc[closeset_idx]['error'] )
        
#     (demand_error_df['error'] - error_noise_df['error'])
    demand_error_df['variance_approx'] = noise_collector
    demand_error_df['error_approx'] = error_collector
    
    return demand_error_df

###
# input format requirements:
# demand curve is a dataframe with column 'demand' and 'error'
# value curve 'value' and 'error'
###     
def calc_price_accu(demand_error_curve, value_error_curve, variance_error_curve,
                    dataset_name, model_type, kernel_type, DATABASE, UPLOAD_FOLDER):# demand vs error and value vs error, not NCP or 1/NCP
    demand_noise_curve = find_closest(demand_error_curve, variance_error_curve)
    value_noise_curve = find_closest(value_error_curve, variance_error_curve)
    
    demand_noise_curve = demand_noise_curve.sort_values('variance_approx', ascending=False).reset_index()
    value_noise_curve = value_noise_curve.sort_values('variance_approx', ascending=False).reset_index()
    dist_rho = np.matrix(demand_noise_curve[['demand']].values)
    price_up = np.matrix(value_noise_curve[['value']].values)
    
    demand_noise_curve.to_csv('Test.csv')
    print(demand_noise_curve.variance_approx)
    
    
    one_over_ncp = 1 / np.matrix(demand_noise_curve[['variance_approx']].values)
    print("ooncp:", one_over_ncp)
    price, opt_revenue = MBP_CanModel_DP(one_over_ncp, price_up, dist_rho)
    
    
    price_error_curve = demand_noise_curve[['error', 'error_approx', 'variance_approx']]
#    length_n = len(price_error_curve.index)
    price_error_curve = price_error_curve.assign(price = price )
    
    price_error_curve = price_error_curve.assign(datasetname = dataset_name)
    price_error_curve = price_error_curve.assign(modeltype = model_type)
    price_error_curve = price_error_curve.assign(kerneltype = kernel_type)
    
    price_error_curve.to_sql('price_error_curve_{:s}_{:s}_{:s}'.format(dataset_name, model_type, kernel_type), DATABASE, if_exists='replace')
    DATABASE.commit()
    return price_error_curve, demand_noise_curve, value_noise_curve
    # first, convert demand error to demand variance
    
#%%    
if __name__ == '__main__':
    # Basic Settings
    #%%
    DATABASE_PATH = 'database1.db'
    DATABASE = sqlite3.connect(DATABASE_PATH)
#    DATABASE.text_factory = str
#    DATABASE_INIT_FILE = './testupload/schema.sql'
    UPLOAD_DIR = 'uploads'
#    ALLOWED_EXTENSIONS = set(['csv', 'dump'])
    
    #%%
    # TEST DATA
    # Train data and label
#    trainX = genfromtxt('../dataset/SimulatedLR/TrainX.csv', delimiter=',')
#    trainY = genfromtxt('../dataset/SimulatedLR/TrainY.csv', delimiter=',')
#    # Test data and label
#    TestX = genfromtxt('../dataset/SimulatedLR/TestX.csv', delimiter=',')
#    TestY = genfromtxt('../dataset/SimulatedLR/TestY.csv', delimiter=',')
#    trainX = genfromtxt('../Nimbus/dataset/Simulated/TrainX.csv', delimiter=',')
#    trainY = genfromtxt('../Nimbus/dataset/Simulated/TrainY.csv', delimiter=',')
#    # Test data and label
#    TestX = genfromtxt('../Nimbus/dataset/Simulated/TestX.csv', delimiter=',')
#    TestY = genfromtxt('../Nimbus/dataset/Simulated/TestY.csv', delimiter=',')
#    trainX = genfromtxt('../Nimbus/dataset/CASP/TrainX.csv', delimiter=',')
#    trainY = genfromtxt('../Nimbus/dataset/CASP/TrainY.csv', delimiter=',')
#    # Test data and label
#    TestX = genfromtxt('../Nimbus/dataset/CASP/TestX.csv', delimiter=',')
#    TestY = genfromtxt('../Nimbus/dataset/CASP/TestY.csv', delimiter=',')
    
    trainX = genfromtxt('../Nimbus/dataset/YearMSD/TrainX.csv', delimiter=',')
    trainY = genfromtxt('../Nimbus/dataset/YearMSD/TrainY.csv', delimiter=',')
    # Test data and label
    TestX = genfromtxt('../Nimbus/dataset/YearMSD/TestX.csv', delimiter=',')
    TestY = genfromtxt('../Nimbus/dataset/YearMSD/TestY.csv', delimiter=',')
    #%%
    # Calculate optimal model
#    optimal_model = calc_optimal_model(LogisticRegression, 'LogisticRegression', 'Default', 'teatall', #'testall',typo 
#                       trainX, trainY, TestX, TestY, DATABASE, UPLOAD_DIR,
#                       filetype = 'bestmodel', username = '')
    optimal_model = calc_optimal_model(LinearRegression, 'LinearRegression', 'Default', 'testall1', #'testall',typo 
                   trainX, trainY, TestX, TestY, DATABASE, UPLOAD_DIR,
                   filetype = 'bestmodel', username = '')
    #%% 
#    demand_error_curve = pd.read_csv('../Nimbus/dataset/Simulated/demand_lin_tuned2.csv')
#    value_error_curve =pd.read_csv('../Nimbus/dataset/Simulated/value_lin_tuned.csv')
#    demand_error_curve = pd.read_csv('../Nimbus/dataset/CASP/demand_lin_tuned2.csv')
#    value_error_curve =pd.read_csv('../Nimbus/dataset/CASP/value_lin_tuned.csv')
    
    demand_error_curve = pd.read_csv('../Nimbus/dataset/YearMSD/demand_lin_tuned2.csv')
    value_error_curve =pd.read_csv('../Nimbus/dataset/YearMSD/value_lin_tuned.csv')
    
    #%%
    # retrieve optimal model
    ret_opt_model_path = '''
        select modelpath
        from bestmodel
        where datasetname=? and modeltype=? and kerneltype=?
        '''
    cursor = DATABASE.execute(ret_opt_model_path, ('testall1', 'LinearRegression', 'Default') )
    (opt_model_path, ) = cursor.fetchone()
    print(opt_model_path)
    variances = np.arange(1, 1001)/100.0
    if opt_model_path != None:
        filepath = os.path.join(UPLOAD_DIR, 'bestmodel', opt_model_path)
#        optimal_model = pickle.load( open( filepath , "rb" ) )
                
#        noise_err_curve = calc_error_noise_curve(LogisticRegression, 'LogisticRegression', 'Default', 'testall', #'testall',typo 
#                       trainX, trainY, TestX, TestY, 
#                       optimal_model, variances, DATABASE, UPLOAD_DIR)
        noise_err_curve = calc_error_noise_curve(LinearRegression, 'LinearRegression', 'Default', 'testall1', #'testall',typo 
                       trainX, trainY, TestX, TestY, 
                       optimal_model, variances, DATABASE, UPLOAD_DIR)

    else:
        print('No file retrieved. ')
        
    show_db(DATABASE_PATH)
    DATABASE.commit()
    #%%
    
            
    price_error_curve, demand_noise_curve, value_noise_curve = calc_price_accu(demand_error_curve, value_error_curve, noise_err_curve, 
                                        'testall', 'LinearRegression', 'Default', DATABASE, UPLOAD_DIR)
    DATABASE.commit()
    #%%
    plt.plot(noise_err_curve.variance, noise_err_curve.error)
#    plt.plot(noise_err_curve.error, noise_err_curve.error)
    #%%
#    price_error_curve.plot('error_approx', 'price')
    plt.plot( 1/price_error_curve.variance_approx.values, price_error_curve.price)
    #%%
#    plt.plot( price_error_curve.error_approx.values, price_error_curve.price)
    plt.plot( demand_noise_curve.error_approx.values, price_error_curve.price)
    #%%
#    DATABASE.close()
    plt.plot(1/demand_noise_curve.variance_approx.values, demand_noise_curve.demand.values)
    #%%
    plt.plot(1/demand_noise_curve.variance_approx.values, value_noise_curve.value.values)
#    plt.plot(demand_noise_curve.variance_approx.values, value_noise_curve.value.values)
    #%%
    trX = np.trace(trainX.T.dot(trainX))
    X2norm = trX/trainX.shape[0]
    #%%
    myres = sum((optimal_model.predict(TestX) - TestY) ** 2)/TestY.shape[0]
    myres = sum((optimal_model.predict(trainX) - trainY) ** 2)/trainY.shape[0]