import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.chdir("D:\Imarticus\Python_program")
os.getcwd()

Tel_Data = pd.read_csv("Telecom_Churn.csv")
Tel_Data.isnull().sum()

Tel_Data.drop(["customerID"], axis = 1, inplace=True)

import numpy as np

Tel_Data['Churn'] = np.where(Tel_Data['Churn'] == 'Yes',1,0)

Telcom_Data = pd.get_dummies(Tel_Data)

from sklearn.model_selection import train_test_split
Trainset , Testset = train_test_split(Telcom_Data, train_size = 0.7)

Train_x = Trainset.drop(['Churn'], axis = 1).copy()
Train_y = Trainset['Churn'].copy()
Test_x = Testset.drop(['Churn'], axis = 1).copy()
Test_y=Testset['Churn'].copy()

############### Decision tree ################

from sklearn.tree import DecisionTreeClassifier

M1 = DecisionTreeClassifier(random_state = 123).fit(Train_x,Train_y)

######## Setting up path for visualization 

import pydotplus
from sklearn.tree import export_graphviz

dot_data =export_graphviz(M1, out_file = None, feature_names = Train_x.columns)

graph =pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("Churn_DT_Plot.pdf")

from sklearn.metrics import confusion_matrix

Test_pred = M1.predict(Test_x)

Confusion_Mat = confusion_matrix(Test_y,Test_pred)
sum(np.diagonal(Confusion_Mat))/Test_x.shape[0]*100 #83.65 

from sklearn.metrics import precision_score, recall_score, f1_score

recall_score(Test_y, Test_pred) #78.74
precision_score(Test_y,Test_pred) #85.36
f1_score(Test_y,Test_pred) #81.92
Confusion_Mat[0][1]/sum(Confusion_Mat[0])  #0.11 #FPR

M2 = DecisionTreeClassifier(random_state=123, min_samples_leaf = 100).fit(Train_x,Train_y) 

dot_data =export_graphviz(M2, out_file = None, feature_names = Train_x.columns)
graph =pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("Churn_DT_Plot2.pdf")

#################### Random Forest #################

from sklearn.ensemble import RandomForestClassifier

M1_RF = RandomForestClassifier(random_state = 123).fit(Train_x,Train_y)
Test_pred1 = M1_RF.predict(Test_x)

Confusion_mat_RF=confusion_matrix(Test_y,Test_pred1)
sum(np.diagonal(Confusion_mat_RF))/Test_x.shape[0]*100  #85.57

from sklearn.metrics import precision_score,recall_score,f1_score

recall_score(Test_y,Test_pred1) #78.46
precision_score(Test_y,Test_pred1) #89.57
f1_score(Test_y,Test_pred1) #83.64
Confusion_mat_RF[0][1]/sum(Confusion_mat_RF[0])  #0.0811

Var_importance_Df = pd.concat([pd.DataFrame(M1_RF.feature_importances_),\
                               pd.DataFrame(Train_x.columns)],axis = 1)

Var_importance_Df
Var_importance_Df.columns = ["Avg_Decrese_Gini","Variable_Name"]

M2_RF = RandomForestClassifier(random_state=123, n_estimators = 25,max_features=5,min_samples_leaf=500).fit(Train_x,Train_y)
Test_pred2 = M2_RF.predict(Test_x)

Confusion_mat2_RF = confusion_matrix(Test_y,Test_pred2)
sum(np.diagonal(Confusion_mat2_RF))/Test_x.shape[0] #0.71

recall_score(Test_y,Test_pred2) #0.6465
precision_score(Test_y,Test_pred2) #0.7239
f1_score(Test_y,Test_pred2)  #0.623
Confusion_mat2_RF[0][1]/sum(Confusion_mat2_RF[0]) #0.21

nestimators = [25,50,75]
maxfeatures = [5,7,9]
minsamples = [200,300]
GirdModel_Df = pd.DataFrame()
Grid2_Df = pd.DataFrame()
Grid_Model_Df = pd.DataFrame()
count = 0

Tree_List = []
Num_Feature_List = []
Sample_List = []
Accuracy_List = []

for i in nestimators:          #range(25,100,25)
    for j in maxfeatures:
        for k in minsamples:
            Model_RF = RandomForestClassifier(random_state = 123, n_estimators=i, max_features=j,min_samples_leaf=k).fit(Train_x,Train_y)
            RF_predict = Model_RF.predict(Test_x)
            Confusion_matrix = confusion_matrix(Test_y,Test_pred2)
            accuracy = sum(np.diagonal(Confusion_matrix))/Test_x.shape[0]
            count = count+1
                        
#            Alternate 1
            Tree_List.append(i)
            Num_Feature_List.append(j)
            Sample_List.append(k)
            Accuracy_List.append(accuracy)
            Grid_Model_Df = pd.DataFrame({"Trees" : Tree_List,"Max_features" : Num_Feature_List,\
                                          "Sample" : Sample_List,"Accuracy" : Accuracy_List})           
#            Alternate 2
            temp_Df = pd.DataFrame([[i,j,k,accuracy]])
            GirdModel_Df = GirdModel_Df.append(temp_Df)
            
#            Alternate 3
            temp_Df1 = pd.DataFrame([[i,j,k,accuracy]])
            Grid2_Df = pd.concat([Grid2_Df,temp_Df1],axis=0)

from sklearn.model_selection import GridSearchCV

my_param_grid = {'n_estimators':[25,50,75],
                 'max_features':[5,7,9],
                 'min_samples_leaf':[100,200]}

Grid_Search_Model = GridSearchCV(estimator=RandomForestClassifier(random_state = 123),
                                 param_grid=my_param_grid,
                                 scoring='accuracy',
                                 cv=5).fit(Train_x,Train_y)

Grid_Search_Df = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)

Final_Model = RandomForestClassifier(random_state = 123,n_estimators=75, max_features = 9, min_samples_leaf=100).fit(Train_x,Train_y)
Final_Model_Predict = Final_Model.predict(Test_x)
Confusion_Matrix_Final = confusion_matrix(Test_y,Final_Model_Predict)            
sum(np.diagonal(Confusion_Matrix_Final))/Test_x.shape[0]
