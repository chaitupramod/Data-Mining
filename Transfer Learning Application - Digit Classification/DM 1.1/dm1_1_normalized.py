import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import os
import ast
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
import graphviz
from collections import OrderedDict



'''
This function is used to plot the co-ordinates of pen digits in figures.
'''

def plot_pen_digits_images():
    
    #plotting for a sample of 100 digits
    header_list = ["dim1","dim2","dim3","dim4","dim5","dim6","dim7","dim8","dim9","dim10","dim11","dim12","dim13","dim14","dim15","dim16","label"]
    df = pd.read_csv("datasets/pendigits_train.csv",names=header_list,header=None,nrows=100)  
    df['combined']= df.iloc[:,:-1].values.tolist()
    df["x_co_ords"]=""
    df["y_co_ords"]=""

    combined_list=list(df['combined'])

    for i in range(0,len(combined_list)):
        list_temp = combined_list[i]
        x_list = list_temp[::2]
        y_list = list_temp[1::2]
        df.loc[i,"x_co_ords"] = str(x_list)
        df.loc[i,"y_co_ords"]=str(y_list)
        
    df["indexCol"] = df.index
    dir_name="pendigits_images"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    for i in range(0,len(df)):
    
        plt.figure()
        x= ast.literal_eval(df["x_co_ords"][i])
        y=ast.literal_eval(df["y_co_ords"][i])
        plt.plot(x,y,'-o')
        plt.savefig(dir_name+"/"+"digit_"+str(df["label"][i])+"_index_"+str(df["indexCol"][i])+".png")
        plt.close()




'''
This function is used to perform data pre-processing
'''

def data_preprocessing():

    header_list = ["dim1","dim2","dim3","dim4","dim5","dim6","dim7","dim8","dim9","dim10","dim11","dim12","dim13","dim14","dim15","dim16","label"]
    df = pd.read_csv("datasets/pendigits_train.csv",names=header_list,header=None)
    df_test = pd.read_csv("datasets/pendigits_test.csv",names=header_list,header=None)

    #splitting into train and test
    train_label = df["label"]
    test_label = df_test["label"]
    df = df.iloc[:,:-1]
    df_test = df_test.iloc[:,:-1]

    #Normalizing values to 0-1 scale
    min_max_scaler = preprocessing.MinMaxScaler()
    df = pd.DataFrame(min_max_scaler.fit_transform(df),columns=header_list[:-1])
    df_test = pd.DataFrame(min_max_scaler.fit_transform(df_test),columns=header_list[:-1])
    df["label"] = train_label
    df_test["label"] = test_label
    
    return(df,df_test,train_label,test_label)




'''
This function is used to perform data analysis
'''

def data_analysis(df,df_test):


    #class distribution
    class_distribution_train = dict((df['label'].value_counts()))
    class_distribution_train = dict(sorted(class_distribution_train.items()))
    print(class_distribution_train)
    
    class_distribution_test = dict((df_test['label'].value_counts()))
    class_distribution_test = dict(sorted(class_distribution_test.items()))
    print(class_distribution_test)
    
    print("-"*20+"CLASS DISTRIBUTION - TRAIN "+"-"*20)
    for key,value in class_distribution_train.items():
        print(key,value)
    
    print("-"*20+"CLASS DISTRIBUTION - TEST "+"-"*20)
    for key,value in class_distribution_test.items():
        print(key,value)
    print("-"*70)
    

    #dimensionality reduction to visualize data
    tsne_x = np.array(df.iloc[:,1:])
    tsne_co_ods = TSNE(n_components=2).fit_transform(tsne_x)
    
    x_tsne = tsne_co_ods[:,0]
    y_tsne = tsne_co_ods[:,1]
    labels = np.array(df['label'])

    color_dict = {"0":"#f58231",
                  "1":"#ffe119",
                  "2":"#bfef45",
                  "3":"#42d4f4",
                  "4":"#000075",
                  "5":"#911eb4",
                  "6":"#f032e6",
                  "7":"#808000",
                  "8":"#000000",
                  "9":"#948B3D",
                  }
    
    
    colors = [color_dict[str(i)] for i in labels]
    

    plt.scatter(x_tsne, y_tsne, c=colors, alpha=0.5)
    plt.title('Digits in high dimensional space as visualized in T-SNE')
    plt.xlabel('x tsne')
    plt.ylabel('y tsne')
    plt.savefig("tsne_visulaization.png")
    plt.show()





'''
This function performs k-fold cross-validation over train set of pen digits data set.
'''

def cross_validation(df):

    diff_model_metrics = {}
    box_plots_accuracy=[]

    criterion_list = ["gini","entropy"]
    for criterion_type in criterion_list:
    
        kf = KFold(n_splits=5)
  
        i = 0
        cvs_accuracy= []
        cvs_macro_avg_precision = []
        cvs_weighted_avg_precision = []
        cvs_macro_avg_recall = []
        cvs_weighted_avg_recall = []
        cvs_macro_avg_f1 = []
        cvs_weighted_avg_f1 = []
        classification_metrics_list = []
        
        for train, test in kf.split(df):
            train_indexes = list(train)
            test_indexes = list(test)
            
            train_set = df.iloc[train_indexes]
            test_set = df.iloc[test_indexes]
            
            i=i+1
            
            X_train = train_set.iloc[:,:-1]
            y_train = train_set.iloc[:,-1]
            X_test = test_set.iloc[:,:-1]
            y_test = test_set.iloc[:,-1]
            
            
            clf = DecisionTreeClassifier(criterion=criterion_type,splitter="best",random_state=19)
            clf.fit(X_train, y_train)
            
            
            y_pred = clf.predict(X_test)
            
            classification_metrics = classification_report(y_test, y_pred,output_dict=True)
            classification_metrics_list.append(classification_metrics)
            
            cvs_accuracy.append(classification_metrics["accuracy"])
            box_plots_accuracy.append(classification_metrics["accuracy"])
            cvs_macro_avg_precision.append(classification_metrics["macro avg"]["precision"])
            cvs_macro_avg_recall.append(classification_metrics["macro avg"]["recall"])
            cvs_macro_avg_f1.append(classification_metrics["macro avg"]["f1-score"])
            cvs_weighted_avg_precision.append(classification_metrics["weighted avg"]["precision"])
            cvs_weighted_avg_recall.append(classification_metrics["weighted avg"]["recall"])
            cvs_weighted_avg_f1.append(classification_metrics["weighted avg"]["f1-score"])
        
        
        mean_cvs_accuracy = statistics.mean(cvs_accuracy)
        mean_cvs_macro_avg_precision = statistics.mean(cvs_macro_avg_precision)
        mean_cvs_macro_avg_recall = statistics.mean(cvs_macro_avg_recall)
        mean_cvs_macro_avg_f1 = statistics.mean(cvs_macro_avg_f1)
        mean_cvs_weighted_avg_precision = statistics.mean(cvs_weighted_avg_precision)
        mean_cvs_weighted_avg_recall = statistics.mean(cvs_weighted_avg_recall)
        mean_cvs_weighted_avg_f1 = statistics.mean(cvs_weighted_avg_f1)
    
             
        keys_list = ["mean_cvs_accuracy","mean_cvs_macro_avg_precision","mean_cvs_macro_avg_recall","mean_cvs_macro_avg_f1",
                   "mean_cvs_weighted_avg_precision","mean_cvs_weighted_avg_recall","mean_cvs_weighted_avg_f1"]
     
        values_list = [mean_cvs_accuracy,mean_cvs_accuracy,mean_cvs_macro_avg_precision,mean_cvs_macro_avg_recall,mean_cvs_macro_avg_f1,
                   mean_cvs_weighted_avg_precision,mean_cvs_weighted_avg_recall,mean_cvs_weighted_avg_f1]
             
        diff_model_metrics[criterion_type] = dict(zip(keys_list,values_list))
        
    return(diff_model_metrics,box_plots_accuracy)




'''
This function plots box plots and also prints classification report on the k-fold cross validations.
'''

def plot_metrics_and_model_selection(diff_model_metrics,box_plots_accuracy):

    n_groups = 7
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    
    plt.bar(index, diff_model_metrics["gini"].values(), bar_width,
                     color='g',
                     label='gini')
    
    plt.bar(index + bar_width,diff_model_metrics["entropy"].values(), bar_width,
                     color='b',
                     label='entropy')

    plt.xlabel('Metrics (AVG)')
    plt.ylabel('Scores')
    plt.title('Avg scores of models during Cross Validation')
    plt.xticks(index + bar_width, ["acc","precission","recall","f1","wt precission","wt recall","wt f1"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("models_cvs_avg_metrics.png")
    plt.show()
    
    for key, value in diff_model_metrics.items():
        print("------------------ Split-Criterion ------------------")
        print(key)
        for key1,value1 in value.items():
            print(key1+"\t:"+str(value1))
    
    
    gini_accuracy = box_plots_accuracy[0:5]
    entropy_accuracy = box_plots_accuracy[5:]
    
    box_plot_df = pd.DataFrame()
    box_plot_df["kfolds"]= [1,2,3,4,5,1,2,3,4,5]
    box_plot_df["split_criterion"] = "gini"
    box_plot_df.iloc[5:10,1]="entropy"
    box_plot_df["accuracy"] = ""
    box_plot_df.iloc[0:5,2] = gini_accuracy
    box_plot_df.iloc[5:10,2] = entropy_accuracy
    
    box_plot_df.boxplot(column='accuracy', by='split_criterion')
    
    
    
'''
This function performs final training of the model with whole train set.
'''

def model_training_and_testing(df):

    clf_final = DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=19)
    tot_X_train = df.iloc[:,:-1]
    tot_y_train = df.iloc[:,-1]
    tot_X_test = df_test.iloc[:,:-1]
    tot_y_test = df_test.iloc[:,-1]
    
    clf_final.fit(tot_X_train, tot_y_train)
    y_predict = clf_final.predict(tot_X_test)
    report = classification_report(tot_y_test, y_predict)
    print("\n")
    print("-"*20+"CLASSIFICATION REPORT"+"-"*20)
    print(report)
    return(clf_final)




'''
This function saves trained model to a file.
'''

def save_model(clf_final):
    #saving model to file
    filename = 'models/dm1_1model.sav'
    pickle.dump(clf_final, open(filename, 'wb'))


'''
This function prints the features importances as determined by the trained model.
'''

def feature_importances(clf_final,df):
    
    feature_importances = dict(zip(df.columns, clf_final.feature_importances_))    
    feature_importances = OrderedDict(sorted(feature_importances.items(), key=lambda feature_importances: feature_importances[1], reverse=True))
    print("-"*20+"FEATURE IMPORTANCES"+"-"*20)
    for key,value in feature_importances.items():
        print(key,value)
    print("-"*60)




'''
This function plots the learning curves which is used to determine the model performance.
'''

def plot_learning_curves(df,train_label):
    
    train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=19), df, train_label, train_sizes=[10, 20, 30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,250,300], cv=5, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    
    plt.figure()
    plt.title("DecisionTreeClassifier")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.ylim(-.1,1.1)
    plt.savefig("learning_curves.png")
    plt.show()




'''
This function plots the final decision tree and prints it to a pdf file.
'''
 
def draw_decision_tree(clf_final,header_list):

    dot_data = tree.export_graphviz(clf_final, out_file=None, feature_names=header_list[:-1],filled=True, rounded=True,special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")
    
 
    

df,df_test,train_label,test_label = data_preprocessing()
#data_analysis(df,df_test)
diff_model_metrics,box_plots_accuracy = cross_validation(df)
plot_metrics_and_model_selection(diff_model_metrics,box_plots_accuracy)
clf_final = model_training_and_testing(df)
feature_importances(clf_final,df)
plot_learning_curves(df,train_label)
save_model(clf_final)
#draw_decision_tree(clf_final,df.columns)




