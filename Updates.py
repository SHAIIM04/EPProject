#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score ,confusion_matrix , classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score , cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from xgboost import XGBClassifier
from sklearn.svm import SVC


# In[2]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_scoreN(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        pred = (pred > 0.5)
        fpr, tpr, _ = metrics.roc_curve(y_train, pred)
        auc_score = metrics.auc(fpr, tpr)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tAUC score: {auc_score * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        print(classification_report(y_train,pred, digits = 3))
        
    elif train==False:
        pred = clf.predict(X_test)
        pred = (pred > 0.5)
        fpr, tpr, _ = metrics.roc_curve(y_test, pred)
        auc_score = metrics.auc(fpr, tpr)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tAUC score: {auc_score * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[3]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        fpr, tpr, _ = metrics.roc_curve(y_train, pred)
        auc_score = metrics.auc(fpr, tpr)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tAUC score: {auc_score * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        print(classification_report(y_train,pred, digits = 3))
        
    elif train==False:
        pred = clf.predict(X_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, pred)
        auc_score = metrics.auc(fpr, tpr)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tAUC score: {auc_score * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        print(classification_report(y_test,pred, digits = 3))


# In[4]:


data = pd.read_csv('train_LZdllcl (2).csv')


# In[ ]:





# In[5]:


data.describe()


# In[6]:


data.head(100)


# In[7]:


data.isnull().values.any()


# In[8]:


categorical_col = []
for column in data.columns:
    if data[column].dtype == object and len(data[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"{column} : {data[column].unique()}")
        print("--------------------------------")


# In[9]:


data.education.value_counts()


# In[10]:


# col=['mediumaquamarine', 'lightskyblue']
# sb.set_palette(col)
# sb.countplot('is_promoted', data=data, hatch=["//","x"],edgecolor='black',)
# plt.show()
label1=data[data["is_promoted"]==0]
label2=data[data["is_promoted"]==1]

n_groups = 1
means_frank = label1['is_promoted'].value_counts()
means_guido = label2['is_promoted'].value_counts()

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='mediumaquamarine',
hatch="//",
edgecolor='black',
label='not promoted')

rects2 = plt.bar(index + bar_width , means_guido, bar_width, 
alpha=opacity,
color='lightskyblue',
hatch="x",
edgecolor='black',
label='promoted')

plt.xlabel('is_promoted')
plt.ylabel('Count')
#plt.title('no_of_trainings')
plt.legend()


plt.tight_layout()
plt.show()


# In[11]:


#using one hot encoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
  
#encode department column
enc = OneHotEncoder(sparse=False)
cols_to_convert = ['department'] # specify the list of columns here
department_col = pd.get_dummies(data['department'], prefix='department')
data['department'] = enc.fit_transform(data.loc[:,cols_to_convert])
data = data.join(department_col,how="right") #Join the new columns
data.head()


# In[12]:


data.info()


# In[13]:


data.dropna(inplace=True)


# In[14]:


data.info()


# In[15]:


data.is_promoted.value_counts()


# In[16]:


4668/54807


# In[17]:


#encode education column
enc = OneHotEncoder(sparse=False)
cols_to_convert = ['education'] # specify the list of columns here
education_col = pd.get_dummies(data['education'], prefix='education')
data['education'] = enc.fit_transform(data.loc[:,cols_to_convert])
data = data.join(education_col,how="right") #Join the new columns
data.head()


# In[18]:


#encode recruitment_channel column
enc = OneHotEncoder(sparse=False)
cols_to_convert = ['recruitment_channel'] # specify the list of columns here
recruitment_channel_col = pd.get_dummies(data['recruitment_channel'], prefix='recruitment_channel')
data['recruitment_channel'] = enc.fit_transform(data.loc[:,cols_to_convert])
data = data.join(recruitment_channel_col,how="right") #Join the new columns
data.head()


# In[19]:


data.shape


# In[20]:


data.info()


# In[21]:


data.isnull().sum()


# In[22]:


data.previous_year_rating.value_counts()


# In[23]:


#data['previous_year_rating'] = data['previous_year_rating'].replace(np.nan,0) 


# In[24]:


len(data)


# In[25]:


categorical_col = []
for column in data.columns:
    if data[column].dtype == object and len(data[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"{column} : {data[column].unique()}")
        print("--------------------------------")


# In[26]:


data.head(50)


# # split The data

# In[27]:


df = pd.DataFrame(data)


# In[28]:


df


# In[266]:


s0 = df.is_promoted[df.is_promoted.eq(0)].sample(200).index
s1 = df.is_promoted[df.is_promoted.eq(1)].sample(200).index 


# In[267]:


test = df.loc[s0.union(s1)]


# In[268]:


test


# In[269]:


train = pd.concat([df, test, test]).drop_duplicates(keep=False)


# In[270]:


train


# In[271]:


train.is_promoted.value_counts()


# In[272]:


test.is_promoted.value_counts()


# In[203]:


48660 * 20/100


# In[204]:


9732 * 40/100


# In[205]:


9732 * 60/100


# In[206]:


data.is_promoted.value_counts()


# # select data

# In[273]:


X = data.iloc[:,[2,4,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23, 24, 25, 26,27,28]]
Y = data.iloc[:,13]
print(X.shape) 
print(Y.shape)


# In[274]:


X_train = train.iloc[:,[2,4,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23, 24, 25, 26,27,28]]
Y_train = train.iloc[:,13]
print(X_train.shape) 
print(Y_train.shape)


# In[275]:


X_test = test.iloc[:,[2,4,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23, 24, 25, 26,27,28]]
Y_test = test.iloc[:,13]
print(X_test.shape) 
print(Y_test.shape)


# In[210]:


#X = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]]
#Y = data.iloc[:,-1]
#print(X.shape)
#print(Y.shape)


# In[211]:


train.columns


# In[212]:


train.head()


# In[213]:


#label encoder


# In[276]:


from sklearn.preprocessing import LabelEncoder
# encoder lables 
enc= LabelEncoder()
X.iloc[:,[0,1]]= X.iloc[:,[0,1]].apply(enc.fit_transform)


# In[277]:


from sklearn.preprocessing import LabelEncoder
# encoder lables 
enc= LabelEncoder()
X_train.iloc[:,[0,1]]= X_train.iloc[:,[0,1]].apply(enc.fit_transform)


# In[278]:


from sklearn.preprocessing import LabelEncoder
# encoder lables 
enc= LabelEncoder()
X_test.iloc[:,[0,1]]= X_test.iloc[:,[0,1]].apply(enc.fit_transform)


# In[279]:


X_train.head(20)


# In[280]:


X_train.columns


# In[219]:


Col_name = ['no_of_trainings','age' , 'previous_year_rating' , 'length_of_service' , 'avg_training_score' ]


# In[220]:


X_test.head(20)


# In[ ]:





# # Exp1

# ### split data train , test 

# In[221]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3 , random_state=0 , stratify = Y , shuffle=True)


# In[222]:


#y_train.value_counts()


# In[223]:


#y_test.value_counts()


# # Balancing Data 

# #### Over sampling

# In[224]:


from imblearn.over_sampling import SMOTE


# In[225]:


# Implementation over sampling SMOTAT
smo = SMOTE()


# In[281]:


# X and Y after balance
x_SM,y_SM = smo.fit_sample(X_train,Y_train)


# In[227]:


Y_train.value_counts()


# In[228]:


y_SM.value_counts()


# #### Under sampling

# In[229]:


# Implementation under sampling Tomek


# In[230]:


from imblearn.under_sampling import TomekLinks


# In[282]:


tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X_train, Y_train)


# In[232]:


Y_train.value_counts()


# In[233]:


y_tl.value_counts()


# #### Over and Under sampling 

# In[234]:


from imblearn.combine import SMOTETomek 


# In[283]:


smt = SMOTETomek()
x_SMtl,y_SMtl = smt.fit_sample(X_tl,y_tl)


# In[236]:


y_SMtl.value_counts()


# #### Standard Scaler

# In[237]:


from sklearn.preprocessing import StandardScaler


# In[238]:


#sca = StandardScaler()
#x_SM.iloc[:,[2,3,4,5,8]] = x_SM.iloc[:,[2,3,4,5,8]].apply(sca.fit_transform)


#X_tl.iloc[:,[2,3,4,5,8]] = X_tl.iloc[:,[2,3,4,5,8]].apply(sca.fit_transform)

#x_SMtl.iloc[:,[2,3,4,5,8]] = x_SMtl.iloc[:,[2,3,4,5,8]].apply(sca.fit_transform)

#X_test.iloc[:,[2,3,4,5,8]] = X_test.iloc[:,[2,3,4,5,8]].apply(sca.fit_transform)


# In[239]:


Col_name


# In[ ]:





# In[284]:


sca = StandardScaler()

scale_SM = x_SM[Col_name]
x_SM[Col_name] = sca.fit_transform(scale_SM.values)

scale_tl = X_tl[Col_name]
X_tl[Col_name]= sca.transform(scale_tl.values)
#X = sca.transform(X)

scale_SMtl = x_SMtl[Col_name]
x_SMtl[Col_name] = sca.fit_transform(scale_SMtl.values)



scale_test = X_test[Col_name]
X_test[Col_name] = sca.fit_transform(scale_test.values)


# In[241]:


X_test


# In[242]:


x_SMtl


# In[243]:


print(x_SM)


# In[244]:


print(X_tl)


# # Split the train to train and validation set

# #### Split for Over sampling 

# In[285]:


from sklearn.model_selection import train_test_split
Xc_train, X_cv, yc_train, y_cv = train_test_split(x_SM, y_SM, test_size=0.2 , random_state=1 , stratify = y_SM )


# In[246]:


yc_train.value_counts()


# In[247]:


y_cv.value_counts()


# # Random Forest

# In[248]:


#train


# In[295]:


from sklearn.ensemble import RandomForestClassifier
classifierOvRF = RandomForestClassifier(n_estimators = 150 ,criterion = 'entropy' , random_state = 0)
classifierOvRF.fit(Xc_train,yc_train)


# In[296]:


scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOvRF ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


# test


# In[254]:


y_pred = classifierOvRF.predict(X_test)
classifierOvRF.score(X_test,Y_test)


# In[294]:


print_score(classifierOvRF, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOvRF, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # Desicon Tree

# In[86]:


#train


# In[83]:


classifierOVD = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifierOVD.fit(Xc_train,yc_train)


# In[84]:


#crossV


# In[85]:


scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOVD ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[90]:


#test


# In[86]:


y_pred = classifierOVD.predict(X_test)
classifierOVD.score(X_test,Y_test)


# In[87]:


print_score(classifierOVD, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOVD, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # SVM

# In[88]:


#train


# In[89]:


classifierOVS = SVC(kernel='rbf', random_state = 0)
classifierOVS.fit(Xc_train,yc_train)


# In[90]:


#crossV


# In[91]:


scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOVS ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[92]:


#test


# In[93]:


Y_pred = classifierOVS.predict(X_test)
classifierOVS.score(X_test,Y_test)


# In[94]:


print_score(classifierOVS, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOVS, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,Y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # KNN

# In[100]:


#train


# In[101]:


classifierOVN = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm = 'brute')
classifierOVN.fit(Xc_train,yc_train)


# In[102]:


#CrossV


# In[103]:


scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOVN ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[104]:


#test


# In[105]:


Y_pred = classifierOVN.predict(X_test)
classifierOVN


# In[106]:


print_score(classifierOVN, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOVN, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,Y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ANN

# In[107]:


#train
annO = tf.keras.models.Sequential()
annO.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
annO.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
annO.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
annO.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
annO.fit(Xc_train,yc_train , batch_size = 32, epochs = 30 )


# In[108]:


#CrossV
#CrossV
from keras.wrappers.scikit_learn import KerasClassifier
def create_network():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
    model.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return model



network = KerasClassifier(build_fn=create_network, 
                                 epochs=30, 
                                 batch_size=32
                                 )
    
    
    
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(network ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)

    
    


# In[109]:


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test


# In[111]:


print_scoreN(annO, Xc_train,yc_train, X_test,Y_test, train=True)
print_scoreN(annO, Xc_train,yc_train, X_test,Y_test, train=False)

y_pred = annO.predict(X_test)
y_pred = (y_pred > 0.5)
print(classification_report(Y_test,y_pred, digits = 3))
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # XGboot

# In[112]:


#train 
classifierOXG = XGBClassifier()
classifierOXG.fit(Xc_train,yc_train)


# In[113]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOXG ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[114]:


#test


# In[115]:


y_pred = classifierOXG.predict(X_test)
classifierOXG.score(X_test,Y_test)


# In[116]:


print_score(classifierOXG, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOXG, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# #### Split for imbalace data

# In[117]:


Xc_train, X_cv, yc_train, y_cv = train_test_split(X_train, Y_train, test_size=0.2 , random_state=3, stratify = Y_train )


# # RF

# In[118]:


#train
classifierURF = RandomForestClassifier(n_estimators = 80 ,criterion = 'entropy' , random_state = 0)
classifierURF.fit(Xc_train,yc_train)


# In[119]:


#crossV


# In[120]:


scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierURF ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[121]:


#test


# In[122]:


y_pred = classifierURF.predict(X_test)
classifierURF.score(X_test,Y_test)


# In[123]:


print_score(classifierURF, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierURF, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))
#AUC
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # DT

# In[124]:


#train
classifierUD = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifierUD.fit(Xc_train,yc_train)


# In[125]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUD ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[126]:


#test
y_pred = classifierUD.predict(X_test)
classifierUD.score(X_test,Y_test)


# In[127]:


print_score(classifierUD, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUD, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # SVM

# In[128]:


#train
classifierUS = SVC(kernel='rbf', random_state = 0)
classifierUS.fit(Xc_train,yc_train)


# In[129]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUS ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[130]:


#test
y_pred = classifierUS.predict(X_test)
classifierUS.score(X_test,Y_test)


# In[131]:


print_score(classifierUS, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUS, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # KNN

# In[132]:


#train
classifierUN = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm = 'brute')
classifierUN.fit(Xc_train,yc_train)


# In[133]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUN ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[134]:


#test
y_pred = classifierUN.predict(X_test)
classifierUN.score(X_test,Y_test)


# In[ ]:





# In[135]:


print_score(classifierUN, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUN, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ANN

# In[136]:


#train
annU = tf.keras.models.Sequential()
annU.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
annU.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
annU.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
annU.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
annU.fit(Xc_train,yc_train , batch_size = 32, epochs = 30 )


# In[137]:


#CrossV
from keras.wrappers.scikit_learn import KerasClassifier
def create_network():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
    model.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return model



network = KerasClassifier(build_fn=create_network, 
                                 epochs=30, 
                                 batch_size=32
                                 )
    
    
    
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(network ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)

    
    


# In[138]:


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
print_scoreN(annU, Xc_train,yc_train, X_test,Y_test, train=True)
print_scoreN(annU, Xc_train,yc_train, X_test,Y_test, train=False)

y_pred = annU.predict(X_test)
y_pred = (y_pred > 0.5)
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # XGboost

# In[ ]:


#train 
classifierUXG = XGBClassifier()
classifierUXG.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUXG ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierUXG.predict(X_test)
classifierUXG.score(X_test,Y_test)


# In[ ]:


print_score(classifierUXG, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUXG, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# #### Split for under-sample

# In[ ]:


Xc_train, X_cv, yc_train, y_cv = train_test_split(X_tl, y_tl, test_size=0.2 , random_state=3 , stratify = y_tl)


# # RF

# In[ ]:


classifierUNRF = RandomForestClassifier(n_estimators = 80 ,criterion = 'entropy' , random_state = 0)
classifierUNRF.fit(Xc_train,yc_train)


# In[ ]:


#crossV

#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUNRF ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test


# In[ ]:


y_pred = classifierUNRF.predict(X_test)
classifierUNRF.score(X_test,Y_test)


# In[ ]:


print_score(classifierUNRF, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUNRF, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))
#AUC
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # DT

# In[ ]:


#train
classifierUND = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifierUND.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUND ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierUND.predict(X_test)
classifierUND.score(X_test,Y_test)


# In[ ]:


print_score(classifierUND, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUND, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))
#AUC
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # SVM

# In[ ]:


#train
classifierUNS = SVC(kernel='rbf', random_state = 0)
classifierUNS.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUNS ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierUNS.predict(X_test)
classifierUNS.score(X_test,Y_test)


# In[ ]:


print_score(classifierUNS, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUNS, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # KNN

# In[ ]:


#train
classifierUNN = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm = 'brute')
classifierUNN.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUNN ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierUNN.predict(X_test)
classifierUNN.score(X_test,Y_test)


# In[ ]:


print_score(classifierUNN, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUNN, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ANN

# In[ ]:


#train
annUN = tf.keras.models.Sequential()
annUN.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
annUN.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
annUN.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
annUN.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
annUN.fit(Xc_train,yc_train , batch_size = 32, epochs = 30 )


# In[ ]:


#CrossV
from keras.wrappers.scikit_learn import KerasClassifier
def create_network():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
    model.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return model



network = KerasClassifier(build_fn=create_network, 
                                 epochs=30, 
                                 batch_size=32
                                 )
    
    
    
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(network ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)

    
    


# In[ ]:


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
print_scoreN(annUN, Xc_train,yc_train, X_test,Y_test, train=True)
print_scoreN(annUN, Xc_train,yc_train, X_test,Y_test, train=False)

y_pred = annUN.predict(X_test)
y_pred = (y_pred > 0.5)
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # XGboost

# In[ ]:


#train 
classifierUNXG = XGBClassifier()
classifierUNXG.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUNXG ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierUNXG.predict(X_test)
classifierUNXG.score(X_test,Y_test)


# In[ ]:


print_score(classifierUNXG, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUNXG, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# #### split for under and over sample

# In[ ]:


Xc_train, X_cv, yc_train, y_cv = train_test_split(x_SMtl, y_SMtl, test_size=0.2 , random_state=3 , stratify = y_SMtl)


# # RF

# In[ ]:


#train
classifierOURF = RandomForestClassifier(n_estimators = 90 ,criterion = 'entropy' , random_state = 0)
classifierOURF .fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOURF ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierOURF .predict(X_test)
classifierOURF .score(X_test,Y_test)


# In[ ]:


print_score(classifierOURF, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOURF, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # DT

# In[ ]:


#train
classifierOUD = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifierOUD.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOUD ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierOUD.predict(X_test)
classifierOUD.score(X_test,Y_test)


# In[ ]:


print_score(classifierOUD, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOUD, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # SVM

# In[ ]:


#train
classifierOUS = SVC(kernel='rbf', random_state = 0)
classifierOUS.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOUS ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierOUS.predict(X_test)
classifierOUS.score(X_test,Y_test)


# In[ ]:


print_score(classifierOUS, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOUS, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # KNN

# In[ ]:


#train
classifierOUN = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm = 'brute')
classifierOUN.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierOUN ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierOUN.predict(X_test)
classifierOUN.score(X_test,Y_test)


# In[ ]:


print_score(classifierOUN, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierOUN, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # ANN

# In[ ]:


#train
annO = tf.keras.models.Sequential()
annO.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
annO.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
annO.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
annO.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
annO.fit(Xc_train,yc_train , batch_size = 32, epochs = 30 )


# In[ ]:


#CrossV
from keras.wrappers.scikit_learn import KerasClassifier
def create_network():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu' , input_dim = 24))
    model.add(tf.keras.layers.Dense(units = 16 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
    model.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return model



network = KerasClassifier(build_fn=create_network, 
                                 epochs=30, 
                                 batch_size=32
                                 )
    
    
    
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(network ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)

    
    


# In[ ]:


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
print_scoreN(annO, Xc_train,yc_train, X_test,Y_test, train=True)
print_scoreN(annO, Xc_train,yc_train, X_test,Y_test, train=False)

y_pred = annO.predict(X_test)
y_pred = (y_pred > 0.5)
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # XGboost

# In[ ]:


#train 
classifierUXG = XGBClassifier()
classifierUXG.fit(Xc_train,yc_train)


# In[ ]:


#crossV
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
           'f1_macro': 'f1_macro' 
          ,'roc_auc' : 'roc_auc'}
scores = cross_validate(classifierUXG ,X_cv, y_cv ,scoring=scoring,
                         cv=10, return_train_score=False)


print(scores['test_acc'].mean())
print(scores['test_prec_macro'].mean())
print(scores['test_rec_micro'].mean())
print(scores['test_f1_macro'].mean())
print(scores['test_roc_auc'].mean())


# In[ ]:


#test
y_pred = classifierUXG.predict(X_test)
classifierUXG.score(X_test,Y_test)


# In[ ]:


print_score(classifierUXG, Xc_train, yc_train, X_test, Y_test, train=True)
print_score(classifierUXG, Xc_train, yc_train, X_test, Y_test, train=False)
print(classification_report(Y_test,y_pred, digits = 3))

fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc_score = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue',
label='ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# # The END EXP1

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# solve it to visulation


# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = sca.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sca.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:


#showing the RF


# In[ ]:


from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,6), dpi=800)
tree.plot_tree(classifier.estimators_[0],
               max_depth = 3 ,
               feature_names = X.columns.values, 
               class_names= ['No', 'Yes'],
               filled = True);
fig.savefig('rf_individualtree.png')


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(t_SM, u_SM, test_size=0.3 , random_state=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30 , criterion = 'entropy' , random_state = 0)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
classifier.score(X_test,y_test)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


8427/13349


# In[ ]:


11095 /13308


# # SVM

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train,y_train)


# In[ ]:


Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


# In[ ]:


print_score(classifier, X_train, y_train, X_test, y_test, train=True)
print_score(classifier, X_train, y_train, X_test, y_test, train=False)


# # Decision Tree

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_SM, y_SM, test_size=0.2 , random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25 , random_state=1)


# In[ ]:





# In[ ]:


classifierD = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)


# In[ ]:


classifierD.fit(X_train,y_train)


# In[ ]:


#val


# In[ ]:


y_pred = classifierD.predict(X_val)
classifierD.score(X_val,y_val)


# In[ ]:


print_score(classifierD, X_train, y_train, X_val, y_val, train=True)
print_score(classifierD, X_train, y_train, X_val, y_val, train=False)


# In[ ]:


y_pred = classifierD.predict(X_test)
classifierD.score(X_test,y_test)


# In[ ]:


print_score(classifierD, X_train, y_train, X_test, y_test, train=True)
print_score(classifierD, X_train, y_train, X_test, y_test, train=False)


# # over and under

# In[ ]:





# In[ ]:





# # artifichal neural network

# In[ ]:


import tensorflow as tf


# In[ ]:


tf.__version__


# In[ ]:


ann = tf.keras.models.Sequential()


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 8 , activation = 'relu' , input_dim = 24))


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 8 , activation = 'relu'))


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))


# In[ ]:


ann.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )


# In[ ]:


ann.fit(X_train, y_train , batch_size = 32 , epochs = 100  )


# In[ ]:


pre = 


# In[ ]:


print_scoreN(ann, X_train, y_train, X_test, y_test, train=True)
print_scoreN(ann, X_train, y_train, X_test, y_test, train=False)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_scoreN(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        pred = (pred > 0.5)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        pred = (pred > 0.5)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# # Exp 2 

# In[ ]:


= df.sample(1000).index

