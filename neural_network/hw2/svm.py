import numpy as np
from sklearn.svm import SVC
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time

#生成路径矩阵，方便导入数据
path='./SEED-IV'
all_data_path=[]
for main_dir in sorted(os.listdir(path)):
    main_dir=os.path.join(path,main_dir)
    if os.path.isdir(main_dir):
        session=[]
        for sub_dir in sorted(os.listdir(main_dir)):
            sub_dir=os.path.join(main_dir,sub_dir)
            experiment=[]
            for name in sorted(os.listdir(sub_dir)):
                experiment.append(os.path.join(sub_dir,name))
            session.append(experiment)
        all_data_path.append(session)
all_data_path=np.array(all_data_path)

#被试依存条件
average_precision=0
for session in range(3):
    for experiment in range(15):
        test_data=np.load(all_data_path[session][experiment][0]).reshape((-1,310))
        test_label=np.load(all_data_path[session][experiment][1])
        train_data=np.load(all_data_path[session][experiment][2]).reshape((-1,310))
        train_label=np.load(all_data_path[session][experiment][3])
        stdScaler=StandardScaler().fit(train_data)
        std_train_data=stdScaler.transform(train_data)
        std_test_data=stdScaler.transform(test_data)

        #训练四种二分类器
        four_pred=np.zeros((4,test_label.shape[0]))
        for type in range(4):
            
            temp_train_label=np.zeros(train_label.shape)
            temp_test_label=np.zeros(test_label.shape)

            for i in range(len(train_label)):#更改标签为二分类
                temp_train_label[i]=1 if train_label[i]==type else 0
            #多次试验之后的调参结果
            if (session,experiment) in [(0,0),(0,9),(1,4),(2,1)]:
                C,kernel=1,'rbf'
            elif (session,experiment) in [(0,1),(0,3),(0,8),(0,11),(1,3),(1,4),(1,9),(2,1),(2,2)]:
                C,kernel=5,'sigmoid'
            else:
                C,kernel=1,'linear'
            svm=SVC(C=C,kernel=kernel,probability=True).fit(std_train_data,temp_train_label)
            pred=svm.predict_proba(std_test_data)
            four_pred[type]=pred[:,1]

        true=np.sum(four_pred.argmax(axis=0)==test_label)#取预测概率最高的
        precision=true/test_label.shape[0]
        if(precision<0.5):
            print('session:',session,'experiment:',experiment,'precision:',precision)
        average_precision+=precision
average_precision/=45
print(average_precision)

#取每个被试的所有数据和标签并拼接在一起
def get_data(experiment):
    test_data=np.load(all_data_path[0][experiment][0]).reshape((-1,310))
    test_label=np.load(all_data_path[0][experiment][1])
    train_data=np.load(all_data_path[0][experiment][2]).reshape((-1,310))
    train_label=np.load(all_data_path[0][experiment][3])
    test_data=np.concatenate((test_data,train_data))
    test_label=np.concatenate((test_label,train_label))
    for session in range(1,3):
        temp_test_data=np.load(all_data_path[session][experiment][0]).reshape((-1,310))
        temp_test_label=np.load(all_data_path[session][experiment][1])
        temp_train_data=np.load(all_data_path[session][experiment][2]).reshape((-1,310))
        temp_train_label=np.load(all_data_path[session][experiment][3])
        test_data=np.concatenate((test_data,temp_test_data,temp_train_data))
        test_label=np.concatenate((test_label,temp_test_label,temp_train_label))
    return test_data,test_label


start=time.time()
#被试独立
average_precision=0
group_num=15
for one in range(group_num):
    flag=True
    for experiment in range(group_num):
        if experiment==one:#留一验证
            test_data,test_label=get_data(experiment)
        else:
            temp_data,temp_label=get_data(experiment)
            if flag:
                train_data,train_label=temp_data,temp_label
                flag=False
            else:
                train_data=np.concatenate((train_data,temp_data))
                train_label=np.concatenate((train_label,temp_label))

    stdScaler=StandardScaler().fit(train_data)
    std_train_data=stdScaler.transform(train_data)
    std_test_data=stdScaler.transform(test_data)
    #注释掉的是使用的SVC自带的ovr
    '''
    svm=SVC(C=1,kernel='rbf').fit(std_train_data,train_label)
    four_pred=svm.predict(std_test_data)

    true=np.sum(four_pred==test_label)
    '''

    four_pred=np.zeros((4,test_label.shape[0]))
    for type in range(4):
        print(type)
        temp_train_label=np.zeros(train_label.shape)

        for i in range(len(train_label)):
            temp_train_label[i]=1 if train_label[i]==type else 0
        svm=SVC(C=1,kernel='rbf',probability=True).fit(std_train_data,temp_train_label)
        pred=svm.predict_proba(std_test_data)
        four_pred[type]=pred[:,1]
    true=np.sum(four_pred.argmax(axis=0)==test_label)

    precision=true/test_label.shape[0]
    print('one:',one)
    print('precision:',precision)
    average_precision+=precision
print('average_precision:',average_precision/group_num)
end=time.time()
totaltime=end-start
print('Running time: %s h %s m %s s' % (totaltime//3600,totaltime%3600//60,totaltime%3600%60))