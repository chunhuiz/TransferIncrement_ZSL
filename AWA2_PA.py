import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveRegressor
import warnings
warnings.filterwarnings("ignore")


path = './data/'

classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}

def make_test_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous-01.txt',header=None,sep = ',')
    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[test_classes_flag,:]

def make_train_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous-01.txt',header=None,sep = ',')
    train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
    train_classes_flag = []
    for item in train_classes.iloc[:,0].values.tolist():
        train_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[train_classes_flag,:]

def generate_data(data_mean,data_std,attribute_table,num):
    class_num = data_mean.shape[0]
    feature_num = data_mean.shape[1]
    data_list = []
    label_list = []
    for i in range(class_num):
        data = []
        for j in range(feature_num):
            data.append(list(np.random.normal(data_mean[i,j],abs(data_std[i,j]),num)))
        data = np.row_stack(data).T
        data_list.append(data)   
        label_list+=[attribute_table.iloc[i,:].values]*num
    return np.row_stack(data_list),np.row_stack(label_list)

import cvxpy as cvx
def linear(X1,As,Au,a,b):
    W = cvx.Variable((As.shape[1],Au.shape[1]))
    X = X1*W
    Xmean = X[:,0]
    for i in range(1,X.shape[1]):
        Xmean += X[:,i]
    Xmean /= X.shape[1]
    
  
    item1 = cvx.sum_squares(cvx.norm(As*W-Au,p=2,axis=0))
    item2 = cvx.norm(Xmean-np.ravel(np.mean(X1,axis=1)))**2   
    item3 = cvx.norm1(W)
    
    objective = cvx.Minimize((item1)+(a*item2)+b*item3)
    prob = cvx.Problem(objective)
    prob.solve()
    return W.value

trainlabel = np.load(path+'AWA2_trainlabel.npy')
train_attributelabel = np.load(path+'AWA2_train_continuous_01_attributelabel.npy')

testlabel = np.load(path+'AWA2_testlabel.npy')
test_attributelabel = np.load(path+'AWA2_test_continuous_01_attributelabel.npy')


trainfeatures = np.load(path+'resnet101_trainfeatures.npy')
testfeatures = np.load(path+'resnet101_testfeatures.npy')

# trainfeatures2 = np.load(path+'Google_trainfeatures.npy')
# testfeatures2 = np.load(path+'Google_testfeatures.npy')
#
# trainfeatures = np.column_stack([trainfeatures1,trainfeatures2])
# testfeatures = np.column_stack([testfeatures1,testfeatures2])

vali_index = list(np.random.choice(trainfeatures.shape[0],size=int(0.2*trainfeatures.shape[0]),replace=False))
train_index = list(set(list(np.arange(0,trainfeatures.shape[0],1)))-set(vali_index))

valifeatures = trainfeatures[vali_index]
valilabel = trainlabel[vali_index]
vali_attributelabel = train_attributelabel[vali_index]

trainfeatures = trainfeatures[train_index]
trainlabel = trainlabel[train_index]
train_attributelabel = train_attributelabel[train_index] 
print("=================================Experiment: Ours+IOM+Res.==================================")
print(trainfeatures.shape,trainlabel.shape,train_attributelabel.shape)
print(testfeatures.shape,testlabel.shape,test_attributelabel.shape)
print(valifeatures.shape,valilabel.shape,vali_attributelabel.shape)

train_attributetable = make_train_attributetable()
test_attributetable = make_test_attributetable()

trainfeatures_tabel = pd.DataFrame(trainfeatures)
trainfeatures_tabel['label'] = trainlabel

trainfeature_mean = np.mat(trainfeatures_tabel.groupby('label').mean().values).T
trainfeature_std = np.mat(trainfeatures_tabel.groupby('label').std().values).T



clf_list = []
print("Base training for {} attribute classifiers begin...".format(train_attributelabel.shape[1]))
for i in range(train_attributelabel.shape[1]):
    if (((i+1)%10 == 0) or (i == (train_attributelabel.shape[1]-1))):
        print("{} th classifier is training".format(i+1))
    clf = PassiveAggressiveRegressor(max_iter=5,C=0.01,loss='epsilon_insensitive')
    clf.fit(trainfeatures,train_attributelabel[:,i])
    clf_list.append(clf)
print("Base training for {} attribute classifiers is over！".format(train_attributelabel.shape[1]))

import copy
number_list = [1,3,5,7,9,10,13,15,17,19,20,21,23,25,27,29,30,31,33,35,37,39,40,43,45,47,50,55,60,70,80,90,100,1000,2000,3000,4000,5000]
a = 0.1
b = 1

import time

t1 = time.time()
W = linear(trainfeature_mean,np.mat(train_attributetable.values).T,np.mat(test_attributetable.values).T,a,b)

virtual_testfeature_mean = (trainfeature_mean*W).T
virtual_testfeature_std = np.ones(virtual_testfeature_mean.shape)*0.3

number_lis = []
time_list = []
train_acc_list = []
test_acc_list = []
H_list = []
import datetime
print("Incremental training begin...")
for number in number_list:
    virtual_testfeature,virtual_test_attributelabel = generate_data(virtual_testfeature_mean,
                                                                    virtual_testfeature_std,
                                                                    test_attributetable,number)
    rand_index = np.random.choice(virtual_testfeature.shape[0],virtual_testfeature.shape[0],replace=False)
    virtual_testfeature = virtual_testfeature[rand_index]
    virtual_test_attributelabel = virtual_test_attributelabel[rand_index]

    test_res_list = []
    vali_res_list = []

    starttime = datetime.datetime.now()
    for i in range(virtual_test_attributelabel.shape[1]):
        clf = copy.deepcopy(clf_list[i])
        clf.partial_fit(virtual_testfeature,virtual_test_attributelabel[:,i])
        res = clf.predict(testfeatures)
        test_res_list.append(res)
        res = clf.predict(valifeatures)
        vali_res_list.append(res)
    endtime = datetime.datetime.now()
    
    test_pre_attribute = np.mat(np.column_stack(test_res_list))
    vali_pre_attribute = np.mat(np.column_stack(vali_res_list))

    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous-01.txt',header=None,sep = ',')

    test_label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i,:]
        loc = np.sum(np.square(attribut_bmatrix.values - pre_res),axis=1).argmin()
        test_label_lis.append(attribut_bmatrix.index[loc])

    vali_label_lis = []
    for i in range(vali_pre_attribute.shape[0]):
        pre_res = vali_pre_attribute[i,:]
        loc = np.sum(np.square(attribut_bmatrix.values - pre_res),axis=1).argmin()
        vali_label_lis.append(attribut_bmatrix.index[loc])
    train_acc = accuracy_score(list(valilabel),vali_label_lis)
    test_acc = accuracy_score(list(testlabel),test_label_lis)
    H = (2*train_acc*test_acc)/(test_acc+train_acc)
    print("Incremental samples are {}, S is {}, U is {}, H is {}, training time is {}"
          .format(number,round(train_acc, 4),round(test_acc, 4), round(H, 4), round((endtime - starttime).total_seconds(),4)))

    number_lis.append(number)
    time_list.append((endtime - starttime).total_seconds())
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    H_list.append(H)

#    Res_AWA2_AP = pd.DataFrame(np.mat([number_lis,time_list,train_acc_list,test_acc_list,H_list]))
#    Res_AWA2_AP.to_excel(path+'Res_AWA_PA.xlsx')
































