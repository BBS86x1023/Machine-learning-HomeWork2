# _*_ coding: utf-8 _*_
"""
Spyder Editor

This is a temporary script file
"""


#%%load each level data seperately
import pickle
number = 94
# save each info separately
Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
for i in range(1, number+1):
    with open("C:\\Users\\BBS\\Desktop\\Machine leaning\\MLGame-master\\games\\arkanoid\\log\\"+str(i)+".pickle", "rb") as f1:
        data_list1 = pickle.load(f1)
        for i in range(0, len(data_list1)):
            Frame.append(data_list1[i].frame)
            Status.append(data_list1[i].status)
            Ballposition.append(data_list1[i].ball)
            PlatformPosition.append(data_list1[i].platform)
            Bricks.append(data_list1[i].bricks)

#with open("C:\\Users\\BBS\\Desktop\\Machine leaning\\MLGame-master\\games\\arkanoid\\log\\2019-09-28_21-03-57.pickle", "rb") as f1:
#    data_list1 = pickle.load(f1)

# save each info separately
'''Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
for i in range(0, len(data_list1)):
    Frame.append(data_list1[i].frame)
    Status.append(data_list1[i].status)
    Ballposition.append(data_list1[i].ball)
    PlatformPosition.append(data_list1[i].platform)
    Bricks.append(data_list1[i].bricks)'''

#print(len(Ballposition))
#_____________________________________________________________________________________________________________________________
import numpy as np
PlatX_plus_20 = []
#底盤移動一次是5格，所以要除與5
#當右往左與左往右來的球判斷輸出為1,0,-1，控制底盤左與右
PlatX=np.array(PlatformPosition)[:,0][:,np.newaxis]
for i in range(0,len(PlatX)):
    PlatX_plus_20.append(20)
PlatX_20=np.array(PlatX_plus_20)[:,np.newaxis]
PlatX = PlatX + PlatX_20
print(PlatX)
PlatX_next=PlatX[1:,:]
instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5
#print(PlatX_next)

#球移動方向
BallX_position = np.array(Ballposition)[:,0][:,np.newaxis]
BallX_position_next = BallX_position[1:,:]
Ball_Vx = BallX_position_next - BallX_position[0:len(BallX_position_next),0][:,np.newaxis]
BallY_position = np.array(Ballposition)[:,1][:,np.newaxis]
BallY_position_next = BallY_position[1:,:]
Ball_Vy = BallY_position_next - BallY_position[0:len(BallY_position_next),0][:,np.newaxis]
#Select some features to make x
#x為輸入特徵向量
Ballarray=np.array(Ballposition[:-1])
x=np.hstack((Ballarray,PlatX[0:-1,0][:,np.newaxis],Ball_Vx,Ball_Vy))
#print(Ballarray)
#Select instructions as y
y=instruct

#split train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.1, random_state=1200)
#_____________________________________________________________________________________________________________________________
#%%train your model here
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn=KNeighborsClassifier(n_neighbors=17)

knn.fit(x_train,y_train)

yknn_bef_scaler=knn.predict(x_test)
acc_knn_bef_scaler=accuracy_score(yknn_bef_scaler,y_test)


#-------------------------------------------------------
#正規化
#from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#scaler.fit(x_train)
#x_train_stdnorm=scaler.transform(x_train)
#knn.fit(x_train_stdnorm, y_train)
#x_test_stdnorm=scaler.transform(x_test)
#yknn_aft_scaler=knn.predict(x_test_stdnorm)
#acc_knn_aft_scaler=accuracy_score(yknn_aft_scaler, y_test)

#_____________________________________________________________________________________________________________________________
filename="C:\\Users\\BBS\\Desktop\\Machine leaning\\MLGame-master\\games\\arkanoid\\ml\\knn_test.sav"
pickle.dump(knn, open(filename, 'wb'))