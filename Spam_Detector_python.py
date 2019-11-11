#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


## import dataset
df = pd.read_csv('creditcard.csv')


# In[3]:


df


# In[4]:


## basic info of dataset
df.dtypes


# In[5]:


df['Time'].value_counts()


# In[6]:


df['Class'].value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


"df.plot.hist(by=str, bins=10)"


# In[10]:


x = df.iloc[:,:30]
y = df.iloc[:,30]
x


# In[ ]:





# # Apply Feature Extraction and Selection

# In[11]:


## first doing feature scaling 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x = sc.fit_transform(x) 


# In[12]:


x = pd.DataFrame(x, columns = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"] )


# # Embeded method to check important feature
# 

# In[13]:



from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()


# # PCA Technique For Feature Extraction

# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components=17)
x = pca.fit_transform(x)

explained_variance = pca.explained_variance_ratio_ 


# In[15]:


explained_variance.sum()


# # Dataset split into Training set and Test set 

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state= 42)


# In[18]:


y_train.value_counts()


# In[19]:


y_test.value_counts()


# # Random Forest Model

# In[20]:


## Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=42)
clf.fit(x_train,y_train)


# In[21]:


pred = clf.predict(x_test)


# In[22]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)


# In[23]:


cm


# In[24]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,pred)


# In[25]:


accuracy


# # Support Vector Classifier Model

# In[26]:


from sklearn.svm import SVC
clf_svc = SVC(kernel='rbf',probability=False,random_state=0)
clf_svc.fit(x_train,y_train)


# In[27]:


pred_svc = clf_svc.predict(x_test)


# In[28]:


from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test,pred_svc)


# In[29]:


cm_svc


# In[30]:


from sklearn.metrics import accuracy_score
accuracy_svc = accuracy_score(y_test,pred_svc)


# In[31]:


accuracy_svc


# # Naive Bayes Model

# In[32]:


from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb = clf_nb.fit(x_train,y_train)


# In[33]:


pred_nb = clf_nb.predict(x_test)


# In[34]:


from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test,pred_nb)


# In[35]:


cm_nb


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_nb = accuracy_score(y_test,pred_nb)


# In[37]:


accuracy_nb


# # Logistic Model

# In[38]:


from sklearn.linear_model import LogisticRegression
clf_lg = LogisticRegression(random_state = 42)
clf_lg.fit(x_train,y_train)


# In[39]:


pred_lg = clf_lg.predict(x_test)


# In[40]:


from sklearn.metrics import confusion_matrix
cm_lg = confusion_matrix(y_test,pred_lg)


# In[41]:


cm_lg


# In[42]:


from sklearn.metrics import accuracy_score
accuracy_lg = accuracy_score(y_test,pred_lg)


# In[43]:


accuracy_lg


# # K-NN Model

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_jobs =-1,n_neighbors=7, metric = 'minkowski',p=2)
clf_knn.fit(x_train,y_train)


# In[45]:


pred_knn = clf_knn.predict(x_test)


# In[46]:


from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test,pred_knn)


# In[47]:


cm_knn


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_knn = accuracy_score(y_test,pred_knn)


# In[49]:


accuracy_knn


# # Decision Tree Model

# In[50]:


from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(criterion='entropy',random_state=42)
clf_dt.fit(x_train,y_train)


# In[51]:


pred_dt = clf_dt.predict(x_test)


# In[52]:


from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test,pred_dt)


# In[53]:


cm_dt


# In[54]:


from sklearn.metrics import accuracy_score
accuracy_dt = accuracy_score(y_test,pred_dt)


# In[55]:


accuracy_dt


# # To compare which model is the best through ROC aand AUC curve

# In[56]:


## logistic regression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_predict
y_score = cross_val_predict(clf_lg,x_train,y_train,cv=3,method='predict_proba')
y_prob = y_score[:,1]
# calculate the fpr and tpr for all thresholds of the classification
fpr,tpr,threshold = roc_curve(y_train,y_prob)
roc_auc = auc(fpr, tpr)

def plot_roc_curve(fpr,tpr, label=None):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = label)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plot_roc_curve(fpr,tpr,label='LR')
plt.show()


# In[57]:


# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=42), 
               GaussianNB(), 
               KNeighborsClassifier(n_neighbors=7, metric = 'minkowski',p=2,n_jobs=-1), 
               DecisionTreeClassifier(random_state=42),
               RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=42)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(x_train, y_train)
    yproba = model.predict_proba(x_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# # Models are overfitted therefore using cross validation for preventing overfitting

# # Random Forest

# In[58]:


from sklearn.model_selection import cross_val_score
Cross = cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')


# In[59]:


Cross.mean()


# # Support Vector Classifier

# In[60]:


Cross_svc = cross_val_score(clf_svc,x_train,y_train,n_jobs =-1,cv=10,scoring='accuracy')


# In[61]:


Cross_svc.mean()


# # Navie Bayes

# In[62]:


Cross_nb = cross_val_score(clf_nb,x_train,y_train,cv=10,scoring='accuracy')


# In[63]:


Cross_nb.mean()


# # Logistioc Regression

# In[64]:


Cross_lg = cross_val_score(clf_lg,x_train,y_train,cv=10,scoring='accuracy')


# In[65]:


Cross_lg.mean()


# # K-NN

# In[66]:


Cross_knn = cross_val_score(clf_knn,x_train,y_train,n_jobs =-1,cv=10,scoring='accuracy')


# In[67]:


Cross_knn.mean()


# # Decision Tree

# In[68]:


Cross_dt = cross_val_score(clf_dt,x_train,y_train,cv=10,scoring='accuracy')


# In[69]:


Cross_dt.mean()


# # Random Forest Model Tuning

# # Using Grid Search method for tuning

# In[70]:


from sklearn.model_selection import GridSearchCV
pram_grid = [{'n_estimators':[10,50,100]}]
grid_search = GridSearchCV(clf, pram_grid, cv=5)


# In[71]:


grid_search.fit(x_train,y_train)


# In[72]:


grid_search.best_estimator_


# In[73]:


pram_grid2 = [{'n_estimators':[50,60,70]}]
grid_search2 = GridSearchCV(clf, pram_grid, cv=5)


# In[74]:


grid_search2.fit(x_train,y_train)


# In[75]:


grid_search2.best_estimator_


# In[76]:


pram_grid3 = [{'n_estimators':[65,70,75,80]}]
grid_search3 = GridSearchCV(clf, pram_grid, cv=5)


# In[77]:


grid_search3.fit(x_train,y_train)


# In[78]:


final_rf = grid_search3.best_estimator_


# In[79]:


final_predict_rm = final_rf.predict(x_test)


# In[80]:


final_rm_cm = confusion_matrix(y_test,final_predict_rm)


# In[81]:


final_rm_cm


# In[126]:


final_accuracy_rm = accuracy_score(y_test,final_predict_rm)


# In[127]:


final_accuracy_rm


# # Logistic Regression Model Tuning 

# In[84]:


parm_grid_lg = [{'C':[0.5,1.0,1.5,2.0],'max_iter':[100,150.200,250,300],
                    'penalty':['l2'],
                   'random_state':[42,50,60],'solver':['lbfgs'],'tol':[0.0001,0.0002,0.0003]}]


# In[85]:


grid_search_lg=GridSearchCV(clf_lg, parm_grid_lg,cv=5)


# In[86]:


grid_search_lg.fit(x_train,y_train)


# In[87]:


grid_search_lg.best_params_


# In[88]:


final_lg=grid_search_lg.best_estimator_


# In[89]:


final_predict_lg=final_lg.predict(x_test)


# In[90]:


final_lg_cm = confusion_matrix(y_test,final_predict_lg)


# In[91]:


final_lg_cm


# In[92]:


final_accuracy_lg = accuracy_score(y_test,final_predict_lg)


# In[93]:


final_accuracy_lg


# # K-NN Model Tuning

# In[100]:


parm_grid_knn = [{'n_neighbors':[5,6,7],'p':[1,2,3]}]


# In[101]:


grid_search_knn = GridSearchCV(clf_knn,parm_grid_knn,n_jobs =-1, cv=2)


# In[102]:


grid_search_knn.fit(x_train,y_train)


# In[105]:


final_knn=grid_search_knn.best_estimator_


# In[106]:


final_pred_knn=final_knn.predict(x_test)


# In[107]:


final_knn_cm = confusion_matrix(y_test,final_pred_knn)


# In[108]:


final_knn_cm


# In[109]:


final_knn_accuracy = accuracy_score(y_test,final_pred_knn)


# In[110]:


final_knn_accuracy


# # SVC Model Tuning

# In[97]:


parm_grid_svc = [{'C':[0.5,1.0,2.0], 
    'degree':[2,3,4],
    'kernel':['linear'],  'random_state':[30,35,42]},{'C':[0.5,1.0,2.0], 
    'degree':[2,3,4],
    'kernel':['rbf'], 'random_state':[30,35,42],'gamma':[0.001,0.0001]}]


# In[98]:


grid_search_svc = GridSearchCV(clf_svc,parm_grid_svc,n_jobs =-1,cv=5 )


# In[99]:


grid_search_svc.fit(x_train,y_train)


# In[111]:


final_svc= grid_search_svc.best_estimator_


# In[112]:


final_svc_pred = final_svc.predict(x_test)


# In[113]:


final_svc_cm= confusion_matrix(y_test,final_svc_pred)


# In[114]:


final_svc_cm


# In[115]:


final_svc_accuracy= accuracy_score(y_test,final_svc_pred)


# In[117]:


final_svc_accuracy


# # Compare Models' accuracy to select the best model

# In[130]:


models_accurracy = [["Random Forest", "Logistic Regression", "K-NN", "SVC"],[final_accuracy_rm,final_accuracy_lg,final_knn_accuracy,final_svc_accuracy]]


# In[131]:


Comparision_matrix = pd.DataFrame(data=models_accurracy,)


# In[133]:


Comparision_matrix


# In[134]:


# after comparing models' accuracy, we find that K-NN and Random Forest model have high accuracy in comparison to other models and they have same accuracy percentage , 
# therefore we can select one of them, i prefer as a final model to Random Forest Classifier,
# because it doesn't take too much time to calculate result in comparison to 
# K-NN, it takes time to calculate the result.


# # Save our Model

# In[135]:


from sklearn.externals import joblib 


# In[136]:


joblib.dump(model, 'spam_detector_model')


# In[ ]:




