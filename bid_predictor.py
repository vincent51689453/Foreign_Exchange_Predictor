import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import csv
from network import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import global_share as gs


def load_dataset():
    i = 0
    print("[INFO] Start loading foreign exchange dataset...")
    print("[INFO] It may take some time... Be patient")
    with open(gs.data_path) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if(i>0):
                #By pass first row which contain labels
                gs.raw_data.append(row)
            i += 1
    print("[INFO] Dataset is ready...")
    return (i-2)

def get_bid_price(data_list,counter):
    i = 0
    while(i<counter):
        price = float(data_list[i][1])
        price = round(price,5)
        gs.bid_data.append(price)
        gs.time_line.append(i)
        i+=1
    print("[INFO] Get bid data successfully")

def display_raw_bid_data(data_list,order,enable):
    if(enable==True):
        #print(data_list)
        fig = plt.figure(figsize=(15,8))
        plt.title('Raw Bid Data of EURUSD')
        plt.scatter(order, data_list, color='r')   
        plt.show()



def split_bid_data(num_data):
    #Prepare training and testing dataset
    np.random.seed(1000)

    #Zip two lists together so they can shuffle in the same order
    temp = list(zip(gs.bid_data,gs.time_line))

    np.random.shuffle(temp)

    #Sepearte two lists
    y_data,x_data = zip(*temp)

    #Normalize y_data within 0 - 1
    sc = MinMaxScaler()
    y_data = np.array(y_data)
    y_data_norm = sc.fit_transform(y_data.reshape(-1,1))
    y_data_norm = y_data_norm.reshape(1,-1)
    
    i = 0
    while(i<num_data):
        gs.y_data_ready.append(y_data_norm[0][i])
        i+=1

    #Normalize bid data within 0 - 1
    gs.bid_data = np.array(gs.bid_data)
    bid_data_norm = sc.fit_transform(gs.bid_data.reshape(-1,1))
    bid_data_norm = bid_data_norm.reshape(1,-1)
    
    i = 0
    while(i<num_data):
        gs.bid_data_ready.append(bid_data_norm[0][i])
        i+=1

    training_size = int(num_data*gs.training_fraction)

    gs.x_train = x_data[:training_size]
    gs.x_test = x_data[training_size:]
    gs.y_train = gs.y_data_ready[:training_size]
    gs.y_test = gs.y_data_ready[training_size:]
    
    


    fig = plt.figure(figsize=(15,8))
    plt.title('Data Distribution')
    plt.scatter(gs.time_line, gs.bid_data_ready, c='blue', label='Market')
    plt.scatter(gs.x_train, gs.y_train, c='red', label='train')
    #plt.scatter(x_test, y_test, c='red', label='validation',s = 2)
    plt.legend()
    plt.show()



def main():
    # Load csv file
    gs.total_data = load_dataset()
    print("[INFO] Number of data:",gs.total_data)
  
    # Only fetch out bid data
    get_bid_price(gs.raw_data,gs.total_data)

    # Scatter plot the bid data in order
    display_raw_bid_data(gs.bid_data,gs.time_line,True)

    #Preprocessing
    split_bid_data(gs.total_data)


    #Start AI Analysis
    loss_function = tf.keras.losses.MeanSquaredError()

    #Create Network
    model = Neural_Network()

    # fit the model on the training dataset
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        gs.initial_learning_rate,
        decay_steps=gs.decay_step ,
        decay_rate=gs.decay_factor,
        staircase=True)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss=loss_function, optimizer=opt)
    history = model.fit(gs.x_train, gs.y_train, epochs=gs.epoches, batch_size=gs.batch_size, validation_data=(gs.x_test, gs.y_test), verbose=2)

    print()
    print("History keys are following: ")
    print(history.history.keys())
    print()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'validation loss'], loc='upper left')
    plt.show()


    #Prediction
    y_pred = model.predict(gs.time_line)
    fig = plt.figure(figsize=(15,8))
    plt.title('AI Prediction of EURUSD')
    plt.scatter(gs.time_line, gs.bid_data_ready, c='green', label='Market',s=2)
    plt.scatter(gs.time_line, y_pred, c='red', label='AI-Prediction',s=4)
    plt.legend()
    plt.grid
    plt.show()








if __name__ == "__main__":
    main()