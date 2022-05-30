
from DECISION import decision_tree
from LOGISTIC import logistic__regression
from SVM import support_vector_machine
import pandas as pd
from preprocessing import preprocess

path =  input("Enter Data : ")
df = pd.read_csv(path)
#graph_obj = Graph(df)
#graph_obj.pca()
pre_obj = preprocess(df)
pre_obj.cleaning()
pre_obj.split_fun()
 
decision_obj = decision_tree(pre_obj.x_train, pre_obj.x_test, pre_obj.y_train, pre_obj.y_test)
decision_obj.classification_decision()

svm_obj = support_vector_machine(pre_obj.x_train, pre_obj.x_test,pre_obj. y_train, pre_obj.y_test)
svm_obj.classification_svm()

logistic_obj=logistic__regression(pre_obj.x_train, pre_obj.x_test, pre_obj.y_train, pre_obj.y_test)
logistic_obj.classification_logistic()

def voting_fun():
    big_list=[]
    for x in range(len(svm_obj.y_pred_svm_test)):
        big_list.append(logistic_obj.y_pred_logistic_test[x] +svm_obj.y_pred_svm_test[x] +decision_obj.y_pred_decision_test[x])
    
        if big_list[x] > 1:
            big_list.pop()
            big_list.append("Cancer")
        else:
            big_list.pop()
            big_list.append("Not Cancer")
            
    print (big_list)                   
voting_fun()
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
  

path1 =  input("Enter Data 2 :  ")
df1 = pd.read_csv(path1)

#graph_obj1 = Graph(df1)
#graph_obj1.pca()
pre_obj1 = preprocess(df1)
pre_obj1.cleaning()
pre_obj1.split_fun()

decision_obj1 = decision_tree(pre_obj1.x_train, pre_obj1.x_test, pre_obj1.y_train, pre_obj1.y_test)
decision_obj1.load_decision()

svm_obj1 = support_vector_machine(pre_obj1.x_train, pre_obj1.x_test,pre_obj1. y_train, pre_obj1.y_test)
svm_obj1.load_svm()


logistic_obj1=logistic__regression(pre_obj1.x_train, pre_obj1.x_test, pre_obj1.y_train, pre_obj1.y_test)
logistic_obj1.load_logistic()


def voting_fun2():
    big_list=[]
    for x in range(len(svm_obj1.result)):
        big_list.append(svm_obj1.result[x] + logistic_obj1.result [x] + decision_obj1.result[x])
        
        if big_list[x] == 3 or big_list[x] == 2:
            big_list.pop()
            big_list.append("Cancer")
        else:
            big_list.pop()
            big_list.append("Not Cancer")
            
    print (big_list)                    
voting_fun2()










