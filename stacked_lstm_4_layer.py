
import time
start_time = time.time()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import xml.etree.ElementTree as ET
import urllib
import re
import io
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.svm import SVR
import datetime
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from math import sqrt
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
# be able to save images on server
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas import Series


#constant parameters
TrainingWindow = 69    #window size is found from historical data
PredictionWindow = 4   #prediction window/prediction horizon
time_sampling = 0.000001	#Should be equal to data refreshing time

#normalizing data
def timeseries_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

	  # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	  # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
	  # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)

    return agg


def scale(df):
	# fit scaler
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler1.fit(df)

    # transform data
    df = df.reshape(df.shape[0], df.shape[1])
    df_scaled = scaler.transform(df)

    return scaler1, scaler, df_scaled


# fit an LSTM network to training data
def fit_forecast_lstm(train, test, epochs, neurons, batch_size):

    # split into input and outputs
    train_X1, train_y = train[:, :-2], train[:, -2:]
    test_X1, test_y = test[:, :-2], test[:, -2:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
    test_X = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(neurons , return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM((neurons/2), return_sequences=True))
    model.add(LSTM((neurons/2), return_sequences=True))
    model.add(LSTM((neurons/4)))
    model.add(Dense(2))
    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'mape'])

    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size , validation_data=(test_X, test_y), verbose=0, shuffle=False)
    yhat = model.predict(test_X)
    #model.summary()

    return model, history ,train_X1 ,test_X1 ,train_X ,test_X, train_y ,test_y, yhat


# plot history
def ploting(name , history):
    # summarize history for loss
    plt.plot(history.history['loss'], '.-')
    lines = plt.plot(history.history['val_loss'], '.-')
    plt.title(name + '_loss')
    plt.legend(['train', 'test'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    xvalues = lines[0].get_xdata()
    yvalues = lines[0].get_ydata()

    idx = np.where(yvalues == min(yvalues))

    min_epoch = xvalues[idx[0][0]]

    print('min loss is:', min(yvalues))

    print('min epoch is:', min_epoch)


# invert scaling for forecast and actual
def invert_scale(test_X, test_y, yhat, scaler):
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, :2]
    inv_yhat = inv_yhat.ravel()

    # invert scaling for actual
    inv_y = np.concatenate((test_y, test_X), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, :2]
    inv_y = inv_y.ravel()

    return inv_y, inv_yhat


def time_h_m(df):
    X = []
    for index, row in df.iterrows():
        X.append([index.hour, index.minute])

    return X


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
#find the correlation among data
def data_correlation(df):
    last_var = df[df.columns[-2:]]
    df = df.drop(columns=['var1(t)', 'var2(t)'])
    frames = []
    for column in df:

        correlated_matrix1 = np.corrcoef(last_var.ix[:, 0], df[column])
        correlated_matrix2 = np.corrcoef(last_var.ix[:, 1], df[column])
        if (correlated_matrix1[0][1] >= abs(0.8) ) or (correlated_matrix2[0][1] >= abs(0.8)) or ((correlated_matrix1[0][1] > abs(0.1)) & (correlated_matrix2[0][1] > abs(0.1))) :
            frames.append(df[column])

    frames.append(last_var)
    correlated_df = pd.concat(frames, axis=1)
    return correlated_df


#moving window regression function
def My_Prediction_Method(ID, df , df3, grouped_df):

    # extracting last readings equivalent to window size
    length = len(df)

        ##SPEED##
    df_si = preparing_data(ID, df3, grouped_df)
    df3_si = df_si[length - TrainingWindow:length]
    #print('df3_si',df3_si)

    # setting Date and time column as an index
    df_si1 = df3_si.set_index('Date_UTC')
    new_index_si = pd.to_datetime(df_si1.index, format='%Y-%m-%d %H:%M:%S')  # converting string data to format of datetime
    df_si02 = df_si1.reindex(new_index_si)  # changing index
    df_si2 = pd.DataFrame(df_si02)

    values_si0 = df_si1.values
    reframed_si = timeseries_to_supervised(values_si0, 1, 1)

    # drop columns we don't want to predict
    reframed_si.drop(reframed_si.columns[440:876], axis=1, inplace=True)

    correlated_reframed_si = data_correlation(reframed_si)

    correlated_reframed_si_values = correlated_reframed_si.values
    scaler1_si, scaler_si, df_si1_scaled = scale(correlated_reframed_si_values)

    # split into train and test sets
    train_si, test_si = df_si1_scaled[0:-PredictionWindow], df_si1_scaled[-PredictionWindow:]

    # design network and make a prediction
    lstm_model_si, lstm_history_si, train_si_X1, test_si_X1, train_si_X, test_si_X, train_si_y, test_si_y, yhat_si = fit_forecast_lstm(
        train_si, test_si, 1579, 64, 16)

    # plotting
    #ploting('Speed and Intensity', lstm_history_si)

    # invert scaling for actual and forecast
    inv_yhat_si, inv_y_si = invert_scale(test_si_X1, test_si_y, yhat_si, scaler1_si)

    return inv_y_si, inv_yhat_si


# calculate RMSE and MAPE
def performance_report(actuals_speed_1,actuals_speed_2 , actuals_speed_3,actuals_speed_4,predictions_speed_1 , predictions_speed_2 ,predictions_speed_3 , predictions_speed_4 , actuals_intensity_1 , actuals_intensity_2 , actuals_intensity_3 , actuals_intensity_4 , predictions_intensity_1 ,predictions_intensity_2 , predictions_intensity_3 ,predictions_intensity_4):

    # report performance
    rmse_speed_1 = sqrt(mean_squared_error(actuals_speed_1, predictions_speed_1))
    rmse_speed_2 = sqrt(mean_squared_error(actuals_speed_2, predictions_speed_2))
    rmse_speed_3 = sqrt(mean_squared_error(actuals_speed_3, predictions_speed_3))
    rmse_speed_4 = sqrt(mean_squared_error(actuals_speed_4, predictions_speed_4))

    print('Speed RMSE for 1st PredWin is: %.3f' % rmse_speed_1)
    print('Speed RMSE for 2nd PredWin is: %.3f' % rmse_speed_2)
    print('Speed RMSE for 3rd PredWin is: %.3f' % rmse_speed_3)
    print('Speed RMSE for 4th PredWin is: %.3f' % rmse_speed_4)

    print('.........................................................')

    MAPE_speed_1 = mean_absolute_percentage_error(actuals_speed_1, predictions_speed_1)
    MAPE_speed_2 = mean_absolute_percentage_error(actuals_speed_2, predictions_speed_2)
    MAPE_speed_3 = mean_absolute_percentage_error(actuals_speed_3, predictions_speed_3)
    MAPE_speed_4 = mean_absolute_percentage_error(actuals_speed_4, predictions_speed_4)

    print('Speed MAPE for 1st PredWin is: %.3f' % MAPE_speed_1)
    print('Speed MAPE for 2nd PredWin is: %.3f' % MAPE_speed_2)
    print('Speed MAPE for 3rd PredWin is: %.3f' % MAPE_speed_3)
    print('Speed MAPE for 4th PredWin is: %.3f' % MAPE_speed_4)

    print('.........................................................')

    rmse_intensity_1 = sqrt(mean_squared_error(actuals_intensity_1, predictions_intensity_1))
    rmse_intensity_2 = sqrt(mean_squared_error(actuals_intensity_2, predictions_intensity_2))
    rmse_intensity_3 = sqrt(mean_squared_error(actuals_intensity_3, predictions_intensity_3))
    rmse_intensity_4 = sqrt(mean_squared_error(actuals_intensity_4, predictions_intensity_4))

    print('Intensity RMSE for 1st PredWin is: %.3f' % rmse_intensity_1)
    print('Intensity RMSE for 2nd PredWin is: %.3f' % rmse_intensity_2)
    print('Intensity RMSE for 3rd PredWin is: %.3f' % rmse_intensity_3)
    print('Intensity RMSE for 4th PredWin is: %.3f' % rmse_intensity_4)

    print('.........................................................')


    MAPE_intensity_1 = mean_absolute_percentage_error(actuals_intensity_1, predictions_intensity_1)
    MAPE_intensity_2 = mean_absolute_percentage_error(actuals_intensity_2, predictions_intensity_2)
    MAPE_intensity_3 = mean_absolute_percentage_error(actuals_intensity_3, predictions_intensity_3)
    MAPE_intensity_4 = mean_absolute_percentage_error(actuals_intensity_4, predictions_intensity_4)

    print('Intensity MAPE for 1st PredWin is: %.3f' % MAPE_intensity_1)
    print('Intensity MAPE for 2nd PredWin is: %.3f' % MAPE_intensity_2)
    print('Intensity MAPE for 3rd PredWin is: %.3f' % MAPE_intensity_3)
    print('Intensity MAPE for 4th PredWin is: %.3f' % MAPE_intensity_4)

    print('.........................................................')

    # line plot of observed vs predicted

    plt.subplot(2, 1, 1)
    plt.plot(actuals_speed_1)
    plt.plot(predictions_speed_1)
    plt.title('speed_1')
    plt.legend(['actual', 'predicted'])

    plt.subplot(2, 1, 2)
    plt.plot(actuals_intensity_1)
    plt.plot(predictions_intensity_1)
    plt.title('intensity_1')
    plt.legend(['actual', 'predicted'])

    plt.xlabel('samples num')

    plt.show()






    plt.subplot(2, 1, 1)
    plt.plot(actuals_speed_2)
    plt.plot(predictions_speed_2)
    plt.title('speed_2')
    plt.legend(['actual', 'predicted'])

    plt.subplot(2, 1, 2)
    plt.plot(actuals_intensity_2)
    plt.plot(predictions_intensity_2)
    plt.title('intensity_2')
    plt.legend(['actual', 'predicted'])

    plt.xlabel('samples num')

    plt.show()





    plt.subplot(2, 1, 1)
    plt.plot(actuals_speed_3)
    plt.plot(predictions_speed_3)
    plt.title('speed_3')
    plt.legend(['actual', 'predicted'])

    plt.subplot(2, 1, 2)
    plt.plot(actuals_intensity_3)
    plt.plot(predictions_intensity_3)
    plt.title('intensity_3')
    plt.legend(['actual', 'predicted'])

    plt.xlabel('samples num')

    plt.show()






    plt.subplot(2, 1, 1)
    plt.plot(actuals_speed_4)
    plt.plot(predictions_speed_4)
    plt.title('speed_4')
    plt.legend(['actual', 'predicted'])

    plt.subplot(2, 1, 2)
    plt.plot(actuals_intensity_4)
    plt.plot(predictions_intensity_4)
    plt.title('intensity_4')
    plt.legend(['actual', 'predicted'])

    plt.xlabel('samples num')

    plt.show()



    print("========================================================================================================")




def preparing_data(ID, df3 , grouped_df):
    df3 = df3[['Date_UTC', 'TrafficSpeed','TrafficIntensity', 'TrafficOccupancy']]
    df3 = df3.reset_index(drop=True)

    grouped_df = grouped_df

    unique_id_list = []

    frames = []
    frames.append(df3)
    for name, group in grouped_df:

        if (group.shape[0] == len(df3)) & (name != ID):
            unique_id_list.append(name)
            df_group = group[['TrafficSpeed', 'TrafficIntensity', 'TrafficOccupancy']]
            df_group = df_group.reset_index(drop=True)
            df_group.columns = ['TrafficSpeed' + '_' + str(name) , 'TrafficIntensity' + '_' + str(name) , 'TrafficOccupancy' + '_' + str(name)]
            frames.append(df_group)

    df_concatated = pd.concat(frames, axis=1)
    return df_concatated



#main function
if __name__ == '__main__':

    actuals_speed_1 = []
    actuals_speed_2 = []
    actuals_speed_3 = []
    actuals_speed_4 = []

    predictions_speed_1 = []
    predictions_speed_2 = []
    predictions_speed_3 = []
    predictions_speed_4 = []

    actuals_intensity_1 = []
    actuals_intensity_2 = []
    actuals_intensity_3 = []
    actuals_intensity_4 = []

    predictions_intensity_1 = []
    predictions_intensity_2 = []
    predictions_intensity_3 = []
    predictions_intensity_4 = []

    # while(1):
    ID = 'PM20412'

    df = pd.read_csv('TotalTrafficDataSet.csv')

    df3 = df[df['ID'] == ID]
    print('len df3 is:',len(df3))
    grouped_df = df.groupby('ID')

    df1 = pd.DataFrame()  # defining an empty dataframe

    for index, row in df3.iterrows():

        df1 = df1.append(row)

        if len(df1) >= TrainingWindow:

            actual, predicted = My_Prediction_Method(ID, df1, df3, grouped_df)

            actuals_speed_1.append(actual[0])
            actuals_speed_2.append(actual[2])
            actuals_speed_3.append(actual[4])
            actuals_speed_4.append(actual[6])


            predictions_speed_1.append(predicted[0])
            predictions_speed_2.append(predicted[2])
            predictions_speed_3.append(predicted[4])
            predictions_speed_4.append(predicted[6])

            actuals_intensity_1.append(actual[1])
            actuals_intensity_2.append(actual[3])
            actuals_intensity_3.append(actual[5])
            actuals_intensity_4.append(actual[7])

            predictions_intensity_1.append(predicted[1])
            predictions_intensity_2.append(predicted[3])
            predictions_intensity_3.append(predicted[5])
            predictions_intensity_4.append(predicted[7])

            if len(predictions_speed_1) % 50 == 0:
                print('len of df1 is: ', len(df1))
                #print('............................................')
                with open("Output-corr-all-PM11201.txt", "w") as text_file:
                    print("\n actuals_speed_1:{} ,\n actuals_speed_2:{} ,\n actuals_speed_3:{} ,\n actuals_speed_4:{},\n predictions_speed_1:{} ,\n predictions_speed_2:{} ,\n predictions_speed_3:{} ,\n predictions_speed_4:{} ,\n actuals_intensity_1:{} ,\n actuals_intensity_2:{} ,\n actuals_intensity_3:{} ,\n actuals_intensity_4:{} ,\n predictions_intensity_1:{} ,\n predictions_intensity_2:{} ,\n predictions_intensity_3:{} ,\n predictions_intensity_4:{} "
                        .format( actuals_speed_1,actuals_speed_2 , actuals_speed_3,actuals_speed_4,predictions_speed_1 , predictions_speed_2 ,predictions_speed_3 , predictions_speed_4 , actuals_intensity_1 , actuals_intensity_2 , actuals_intensity_3 , actuals_intensity_4 , predictions_intensity_1 ,predictions_intensity_2 , predictions_intensity_3 ,predictions_intensity_4 ), file=text_file)
            n = 900

            if len(predictions_speed_1) == n:
                performance_report(actuals_speed_1,actuals_speed_2 , actuals_speed_3,actuals_speed_4,predictions_speed_1 , predictions_speed_2 ,predictions_speed_3 , predictions_speed_4 , actuals_intensity_1 , actuals_intensity_2 , actuals_intensity_3 , actuals_intensity_4 , predictions_intensity_1 ,predictions_intensity_2 , predictions_intensity_3 ,predictions_intensity_4)
                execution_time = time.time() - start_time
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                print('execution time is:', execution_time)

                break

        else:
            pass

            # print ("i am sleeping")
        time.sleep(time_sampling)
