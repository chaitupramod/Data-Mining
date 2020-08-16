# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:20:43 2019

@author: chaitupramod
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

'''
This function is used to sort the data set by the digit value from 0 to 9
'''
def sort_dataset():
    
    df = pd.read_csv("datasets/mnist_train.csv")
    df.sort_values(by=['label'],inplace=True)
    df.to_csv("datasets/sampled/sorted_mnist.csv",header=True,index=False)



'''
This function performs sampling 50 rows for each digit from 0 to 9.
'''
def sample_mnist():
    
    df_sorted = pd.read_csv("datasets/sampled/sorted_mnist.csv")
    size = 50        # sample size
    replace = False  # without replacement
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
    df_sorted = df_sorted.groupby('label', as_index=False).apply(fn)
    df_sorted.to_csv("datasets/sampled/sampled_mnist.csv",header=None,index=False)



'''
This function is used to get co-ordinates from the plotted grey scale images.
'''
def get_co_ordinates():

    coord_dict={}
    
    def onclick(event,count,label):

        if event.xdata != None and event.ydata != None:
            temp_str = str(round(event.xdata))+","+str(round(event.ydata))
            print(temp_str)
            
            key_param = str(label)+"_index_"+str(count)
            
            if( key_param in coord_dict.keys()):
                val = coord_dict[str(label)+"_index_"+str(count)]
                val = val+","+temp_str
                coord_dict[str(label)+"_index_"+str(count)] = val
            else:
                coord_dict[str(label)+"_index_"+str(count)] = temp_str
                    
    count=0       
    with open('datasets/sampled/sampled_mnist.csv', 'r') as csv_file:
        for data in csv.reader(csv_file):
               
            print("ROW: ",count)
            label = data[0]
            pixels = data[1:]
            pixels = np.array(pixels, dtype='uint8')
            pixels = pixels.reshape((28, 28))
            plt.title('Label is {label}'.format(label=label))
        
            ax = plt.imshow(pixels, cmap='gray')
            fig = ax.get_figure()
            fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event,count,label)) 

            plt.show()
            plt.pause(5)
            plt.close()
        
            count=count+1
            

    columns_mnist_coords = "dim1,dim2,dim3,dim4,dim5,dim6,dim7,dim8,dim9,dim10,dim11,dim12,dim13,dim14,dim15,dim16,label"    
    mnist_gen_dataset="datasets/generated_co_ordinates_dataset/mnist_generated_dataset.csv"
    
    if(mnist_gen_dataset not in os.listdir()):
        with open(mnist_gen_dataset,"a+") as f:
            f.write(columns_mnist_coords+"\n")
            
            
    with open(mnist_gen_dataset,"a+") as f:
        for i in coord_dict.keys():
            print(i[0])
            f.write(coord_dict[i]+","+str(i[0])+"\n")

    f.close()



'''
This function is used to test the generated data set by sending it through the trained pen digits classifier.
'''
def test_mnist():

    filename="../DM 1.1/models/dm1_1model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    df = pd.read_csv("datasets/generated_co_ordinates_dataset/mnist_generated_dataset.csv")
    labels = df["label"]
    
    df = df.iloc[:,:-1]
    
    #normalization
    df=(df-df.min())/(df.max()-df.min())
    df["label"] = labels
    
    X_test = df.iloc[:,:-1]
    y_actual_test = df["label"]
    y_predicted = loaded_model.predict(X_test)
    report = classification_report(y_actual_test, y_predicted)
    
    print("---------------- EVALUATION REPORT - MNIST (TEST) -----------------")
    print(report)
    print("-"*68)
    print("ACCURACY")
    print(accuracy_score(y_actual_test, y_predicted)*100)
    print("-"*68)
    
    df["predicted"] = y_predicted
    df.to_csv("result/predicted.csv",index=False)
    
    


#sort_dataset()
#sample_mnist()
#get_co_ordinates()
test_mnist()


        
