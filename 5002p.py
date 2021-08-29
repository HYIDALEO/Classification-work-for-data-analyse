
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from scikitplot.estimators import plot_feature_importances
import seaborn as sns

df = pd.read_csv('C://Users/admin/Desktop/semB/5002/111.csv', index_col=0,encoding='gbk')

predictors = ['Unhappy_About','Complexity','Source_Type','Stage','Lead_age','Customer_age']
df = df[df.Unhappy_About!='']

df = df[df.Source_Type !='']#  df.Lead_age !='' and df.Loss_Reason !=''and df.Customer_age !='']
df = df[df.Stage !='']
df = df[df.Lead_age !='']
df = df[df.Customer_age !='']
df.fillna(-5, inplace=True)
df = df[df.Complexity!=-5]
df = df[df.Complexity!='??']
df = df[df.Complexity!='????']
df = df[df.Complexity!='??????']

training_df = df[0:14000]
testing_df = df[14000:18200]

X_train = training_df[predictors].values
X_test = testing_df[predictors].values



Y1_train = np.array([training_df.Complexity])
Y1_test = np.array([testing_df.Complexity])
up = {'Crowding':[],'Gaps in teeth':[],'Straighter teeth':[],'Minor Crooked teeth':[],'Protruding teeth':[]}
up1 = []
up2 = []
up3 = []
up4 = []
up5 = []
comp = []
leag = []

print(len(X_train))
for i in range(len(X_train)):
    if X_train[i][0] == 'Crowding':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_train[i][0] == 'Crowding; ????':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_train[i][0] == 'Crowding; Gaps in teeth':
        up1.append(1)
        up2.append(1)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_train[i][0] == 'Crowding; Gaps in teeth; Straighter teeth':
        up1.append(1)
        up2.append(1)
        up3.append(1)
        up4.append(0)
        up5.append(0)
    elif X_train[i][0] == 'Crowding; Minor Crooked teeth':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_train[i][0] == 'Crowding; Minor Crooked teeth; Gaps in teeth':
        up1.append(1)
        up2.append(1)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_train[i][0] == 'Gaps in teeth':
        up1.append(0)
        up2.append(1)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_train[i][0] == 'Minor Crooked teeth':
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_train[i][0] == 'Minor Crooked teeth; Gaps in teeth':
        up1.append(0)
        up2.append(1)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_train[i][0] == 'Protruding teeth':
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(1)
    elif X_train[i][0] == 'Protruding teeth; Crowding':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(1)
    elif X_train[i][0] == 'Protruding teeth; Crowding; Minor Crooked teeth':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(1)
    elif X_train[i][0] == 'Protruding teeth; Gaps in teeth':
        up1.append(0)
        up2.append(1)
        up3.append(0)
        up4.append(0)
        up5.append(1)
    elif X_train[i][0] == 'Protruding teeth; Minor Crooked teeth':
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(1)
    elif X_train[i][0] == 'Straighter teeth':
        up1.append(0)
        up2.append(0)
        up3.append(1)
        up4.append(0)
        up5.append(0)
    else:
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(0)

st1 = []
st2 = []
st3 = []
st4 = []
st5 = []
st6 = []
st7 = []
st8 = []
st9 = []
st10 = []

print(len(X_train))
for i in range(len(X_train)):
    if X_train[i][2] == '1_KOL_Direct':
        st1.append(1)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '2_KOL_Ads':
        st1.append(0)
        st2.append(1)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '3_Promotion':
        st1.append(0)
        st2.append(0)
        st3.append(1)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '4_Content':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(1)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '5_Direct':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(1)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '6_Marketing Agency':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(1)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '7_Offline':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(1)
        st8.append(0)
        st9.append(0)
    elif X_train[i][2] == '8_unknown':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(1)
        st9.append(0)
    elif X_train[i][2] == '8_Unknown':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(1)
        st9.append(0)
    elif X_train[i][2] == '9_Referral':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(1)
    else:
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)

sg1 = []

for i in range(len(X_train)):
    if X_train[i][3] == 'Closed Won':
        sg1.append(1)
    elif X_train[i][3] == 'Closed Lost':
        sg1.append(0)
    else:
        sg1.append(0)

cg1 = []
cg2 = []
cg3 = []
cg4 = []
cg5 = []
cg6 = []
cg7 = []

for i in range(len(X_train)):
    if X_train[i][5] == '15-25':
        cg1.append(1)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_train[i][5] == '18-25':
        cg1.append(0)
        cg2.append(1)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_train[i][5] == '26-35':
        cg1.append(0)
        cg2.append(0)
        cg3.append(1)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_train[i][5] == '36-45':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(1)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_train[i][5] == '46-55':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(1)
        cg6.append(0)
        cg7.append(0)
    elif X_train[i][5] == 'Above 55':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(1)
        cg7.append(0)
    elif X_train[i][5] == 'Under 18':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(1)
    else:
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)

up = pd.DataFrame({'Complexity':training_df.Complexity.values,'Lead_age':training_df.Lead_age.values,
                   'Crowding':up1,'Gaps in teeth':up2,'Straighter teeth':up3,'Minor Crooked teeth':up4,'Protruding teeth':up5,
                   '1_KOL_Direct':st1,'2_KOL_Ads':st2,'3_Promotion':st3,'4_Content':st4,'5_Direct':st5,
                   '6_Marketing Agency':st6,'7_Offline':st7,'8_Unknown':st8,'9_Referral':st9,
                   'Stage':sg1,
                   '15-25':cg1,'18-25':cg2,'26-35':cg3,'36-45':cg4,'46-55':cg5,'Above 55':cg6,'Under 18':cg7})
print(up)

for i in up.columns:
    sns.set(rc={'figure.figsize': (7, 5)})
    sns.distplot(up[i], bins=30) # Distribution plot
#    plt.title(i)
    plt.show()

correlation_matrix = up.corr().round(2)

sns.heatmap(data=correlation_matrix, cmap='Blues',annot=True)

plt.show()



up = pd.DataFrame({'Complexity':training_df.Complexity.values,'Lead_age':training_df.Lead_age.values,
                   'Crowding':up1,'Gaps in teeth':up2,'Straighter teeth':up3,'Minor Crooked teeth':up4,'Protruding teeth':up5,
                   '1_KOL_Direct':st1,'2_KOL_Ads':st2,'3_Promotion':st3,'4_Content':st4,'5_Direct':st5,
                   '6_Marketing Agency':st6,'7_Offline':st7,'8_Unknown':st8,'9_Referral':st9,
                   '15-25':cg1,'18-25':cg2,'26-35':cg3,'36-45':cg4,'46-55':cg5,'Above 55':cg6,'Under 18':cg7})
print(up)
X_train = up.values

Y_train = np.array(sg1)
#--------------------------------------------------------------------------------------------------------
up1 = []
up2 = []
up3 = []
up4 = []
up5 = []
comp = []
leag = []

print(len(X_test))
for i in range(len(X_test)):
    if X_test[i][0] == 'Crowding':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_test[i][0] == 'Crowding; ????':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_test[i][0] == 'Crowding; Gaps in teeth':
        up1.append(1)
        up2.append(1)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_test[i][0] == 'Crowding; Gaps in teeth; Straighter teeth':
        up1.append(1)
        up2.append(1)
        up3.append(1)
        up4.append(0)
        up5.append(0)
    elif X_test[i][0] == 'Crowding; Minor Crooked teeth':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_test[i][0] == 'Crowding; Minor Crooked teeth; Gaps in teeth':
        up1.append(1)
        up2.append(1)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_test[i][0] == 'Gaps in teeth':
        up1.append(0)
        up2.append(1)
        up3.append(0)
        up4.append(0)
        up5.append(0)
    elif X_test[i][0] == 'Minor Crooked teeth':
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_test[i][0] == 'Minor Crooked teeth; Gaps in teeth':
        up1.append(0)
        up2.append(1)
        up3.append(0)
        up4.append(1)
        up5.append(0)
    elif X_test[i][0] == 'Protruding teeth':
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(1)
    elif X_test[i][0] == 'Protruding teeth; Crowding':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(1)
    elif X_test[i][0] == 'Protruding teeth; Crowding; Minor Crooked teeth':
        up1.append(1)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(1)
    elif X_test[i][0] == 'Protruding teeth; Gaps in teeth':
        up1.append(0)
        up2.append(1)
        up3.append(0)
        up4.append(0)
        up5.append(1)
    elif X_test[i][0] == 'Protruding teeth; Minor Crooked teeth':
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(1)
        up5.append(1)
    elif X_test[i][0] == 'Straighter teeth':
        up1.append(0)
        up2.append(0)
        up3.append(1)
        up4.append(0)
        up5.append(0)
    else:
        up1.append(0)
        up2.append(0)
        up3.append(0)
        up4.append(0)
        up5.append(0)

st1 = []
st2 = []
st3 = []
st4 = []
st5 = []
st6 = []
st7 = []
st8 = []
st9 = []
st10 = []

print(len(X_test))
for i in range(len(X_test)):
    if X_test[i][2] == '1_KOL_Direct':
        st1.append(1)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '2_KOL_Ads':
        st1.append(0)
        st2.append(1)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '3_Promotion':
        st1.append(0)
        st2.append(0)
        st3.append(1)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '4_Content':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(1)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '5_Direct':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(1)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '6_Marketing Agency':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(1)
        st7.append(0)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '7_Offline':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(1)
        st8.append(0)
        st9.append(0)
    elif X_test[i][2] == '8_unknown':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(1)
        st9.append(0)
    elif X_test[i][2] == '8_Unknown':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(1)
        st9.append(0)
    elif X_test[i][2] == '9_Referral':
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(1)
    else:
        st1.append(0)
        st2.append(0)
        st3.append(0)
        st4.append(0)
        st5.append(0)
        st6.append(0)
        st7.append(0)
        st8.append(0)
        st9.append(0)

sg1 = []

for i in range(len(X_test)):
    if X_test[i][3] == 'Closed Won':
        sg1.append(1)
    elif X_test[i][3] == 'Closed Lost':
        sg1.append(0)
    else:
        sg1.append(0)

cg1 = []
cg2 = []
cg3 = []
cg4 = []
cg5 = []
cg6 = []
cg7 = []

for i in range(len(X_test)):
    if X_test[i][5] == '15-25':
        cg1.append(1)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_test[i][5] == '18-25':
        cg1.append(0)
        cg2.append(1)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_test[i][5] == '26-35':
        cg1.append(0)
        cg2.append(0)
        cg3.append(1)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_test[i][5] == '36-45':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(1)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)
    elif X_test[i][5] == '46-55':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(1)
        cg6.append(0)
        cg7.append(0)
    elif X_test[i][5] == 'Above 55':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(1)
        cg7.append(0)
    elif X_test[i][5] == 'Under 18':
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(1)
    else:
        cg1.append(0)
        cg2.append(0)
        cg3.append(0)
        cg4.append(0)
        cg5.append(0)
        cg6.append(0)
        cg7.append(0)



up = pd.DataFrame({'Complexity':testing_df.Complexity.values,'Lead_age':testing_df.Lead_age.values,
                   'Crowding':up1,'Gaps in teeth':up2,'Straighter teeth':up3,'Minor Crooked teeth':up4,'Protruding teeth':up5,
                   '1_KOL_Direct':st1,'2_KOL_Ads':st2,'3_Promotion':st3,'4_Content':st4,'5_Direct':st5,
                   '6_Marketing Agency':st6,'7_Offline':st7,'8_Unknown':st8,'9_Referral':st9,
                   '15-25':cg1,'18-25':cg2,'26-35':cg3,'36-45':cg4,'46-55':cg5,'Above 55':cg6,'Under 18':cg7})
print(up)

X_test = up.values

Y_test = np.array(sg1)

clf = DecisionTreeClassifier(min_samples_split=15, random_state=0)
tree_est = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
pred_accu = np.sum(Y_pred==Y_test)/len(Y_test)
print('The test error of DecisionTree is ',1-pred_accu)

#draw the optimal separating hyperplane with the scatter plot
svc = LinearSVC()
svc.fit(X_train, Y_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
xlim = plt.xlim()
beta = svc.coef_[0]
beta_0 = svc.intercept_[0]
k = -beta[0] / beta[1] #slope
b = - beta_0 / beta[1] #intercept
xx = np.linspace(xlim[0], xlim[1])
yy = k * xx + b
plt.plot(xx, yy)
plt.show()

Y_pred = svc.predict(X_test)
pred_err=1-np.sum(Y_pred==Y_test)/(len(Y_test))
print(pred_err)
print(confusion_matrix(Y_test, Y_pred))


clf_qda = QDA()
clf_qda.fit(X_train, Y_train)
Y_predict = clf_qda.predict(X_test)
print('The test error of QDA is',np.mean(Y_predict==Y_test))

plt.scatter(X_train[Y_train==1][:,0],X_train[Y_train==1][:,1], facecolors='none', edgecolors='r', label='Up')
plt.show()
plt.scatter(X_train[Y_train==0][:,0],X_train[Y_train==0][:,1], facecolors='none', edgecolors='b', label='Down')
plt.show()


