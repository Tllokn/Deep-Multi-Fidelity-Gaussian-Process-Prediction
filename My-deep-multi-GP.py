import numpy as np
import matplotlib as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel as C
import torch
import GPy

'''数据读取与预处理'''
XTest = np.loadtxt('./dataset/x_test_high.txt').reshape(-1,32)
# Xtest=Xtest.mean(axis=1).reshape(-1,1)
YTest= np.loadtxt('./dataset/y_test_high.txt').reshape(-1,1)

X_low=np.loadtxt('./dataset/x_train_low.txt').reshape(-1,32)
# X_low=X_low.mean(axis=1).reshape(-1,1)
X_high=np.loadtxt('./dataset/x_train_high.txt').reshape(-1,32)
# X_high=X_high.mean(axis=1).reshape(-1,1)
Y_low = (np.loadtxt('./dataset/y_train_low.txt')*1e4).reshape(-1,1)
Y_high= np.loadtxt('./dataset/y_train_high.txt').reshape(-1,1)

XTrain=np.vstack((X_low,X_high))
YTrain=np.vstack((Y_low,Y_high))
# XTrain=np.array([X_low,X_high]).T
# YTrain=np.array([Y_low,Y_high]).T

print(XTrain.shape)
print(YTrain.shape)

XTrain = torch.from_numpy(XTrain).float()
YTrain = torch.from_numpy(YTrain).float()
XTest = torch.from_numpy(XTest).float()
YTest = torch.from_numpy(YTest).float()

'''特征提取网络定义'''
DIn=32
Hidden1=100
Hidden2=50
Hidden3=5
Dout=1
EPOCH=30000


GetFeatureNetModel=torch.nn.Sequential(
    torch.nn.Linear(DIn,Hidden1),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden1,Hidden2),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden2,Hidden3),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden3,Dout)
)
learning_rate=1e-3
loss_fn=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.Adam(GetFeatureNetModel.parameters(),lr=learning_rate)

'''高斯过程定义'''
# k1=RBF(1278)
# k2=RBF(21)
# rho=C(0.1,(0.0,1.0))
# k11=k1
# k12=rho*k1
# k21=rho*k1
# k22=rho*rho*k1+k2

kernel=RBF(5,(2,100))
reg=GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=20,alpha=0.2)



'''网络与训练'''
for t in range(0,EPOCH):
    feature=GetFeatureNetModel(XTrain)
    # feature2=feature.detach().numpy()
    # reg.fit(feature2,YTrain)
    # mean,sigma=reg.predict(feature2,return_std=True)
    # mean=np.array(mean.reshape(-1))
    # YTrain=np.array(YTrain.reshape(-1))
    loss=loss_fn(feature,YTrain)
    if t%1000==0:
        print("第",t,"轮：loss:",loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# mean,sigma=reg.predict(Xtest,return_std=True)
#
# print("prediction",mean)
# print("label",Ytest)
result=GetFeatureNetModel(XTest)
print(YTest)
print(result)
