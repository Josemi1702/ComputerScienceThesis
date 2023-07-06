#Basics Python libreries
from os.path import isfile
import warnings

import pandas as pd
import numpy as np
import matplotlib
import datetime
import seaborn as sns
from pathlib import Path
import pickle
import logging



import matplotlib.pyplot as plt

matplotlib.rc('figure', figsize=(15, 5))

#Sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.tsa.seasonal as tsa
import statsmodels.graphics.tsaplots as tsaplots


#Tensor y keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input,Conv1D, Conv2D, Conv3D, GRU, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten
from keras.utils import set_random_seed


#Yahoo Finance: para cargar datos
import yfinance as yf

#Technical Analisis library
import ta




#global NUM_MODELS, NUM_DATASETS
#NUM_DATASETS = 12
#NUM_MODELS_RNN=3


def execution(num_data, num_rnn, rep):
    
    #warnings.filterwarnings("ignore")
    #logging.getLogger('tensorflow').disabled = True
    
    df= select_dataset(num_data)
    train_data = df['2015':'2022']
    #validation_data = df['2021':'2022']
    test_data = df['2022':]
    
    train_data=np.array(train_data)
    #validation_data=np.array(validation_data)
    test_data=np.array(test_data)
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    #validation_data = scaler.transform(validation_data)
    test_data = scaler.transform(test_data)
    
    
    
    #Creación de secuencias de tiempo
    n_input = 100 #Cantidad de datos anteriores
    n_features = len(df.columns)-3
    
    
    
    time_step = 100 #Bloques de 100 datos (100 anteriores por cada salida)
    X_train = [] #Los 100 datos por cada predicción
    Y_train = [] #Aprendizaje supervisado
    Y_train2= []
    Y_train3= []
    aux=[]

    print("---------------------------------------------TRAIN---------------------------------------")
    for i in range(time_step-1,len(train_data)): #Este buvcle está bien hecho, desde la posicion 99 hasta el final
        Y_train.append(train_data[i, 22])
        Y_train2.append(train_data[i,23])
        Y_train3.append(train_data[i,24])

        aux=[]

        for j in range(time_step-1,-1,-1):
            aux.append(train_data[i-j,0:22]) #En la primera ejecucion son del 0-99, luego 1-100 y así hasta el final
            #print(i-j)
        X_train.append(aux) 


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Y_train2= np.array(Y_train2)
    Y_train3= np.array(Y_train3)
    print(X_train.shape)
    print(Y_train.shape)
    print(Y_train2.shape)
    print(Y_train3.shape)


    print("------------------------------------------------------------------------------------")

    #print("---------------------------------------------Validation---------------------------------------")
   # X_validation = [] #Los 100 datos por cada predicción
   # Y_validation = [] #Aprendizaje supervisado
   # Y_validation2= []
   # Y_validation3= []
   # aux=[]
   # for i in range(time_step-1,len(validation_data)): #Este buvcle está bien hecho, desde la posicion 99 hasta el final
   #     Y_validation.append(validation_data[i,22])
   #     Y_validation2.append(validation_data[i,23])
   #     Y_validation3.append(validation_data[i,24])
   #     aux=[]
#
   #     for j in range(time_step-1,-1,-1):
   #         aux.append(validation_data[i-j,0:22]) #En la primera ejecucion son del 0-99, luego 1-100 y así hasta el final
   #         #print(i-j)
   #     X_validation.append(aux) 
#
#
   # X_validation = np.array(X_validation)
   # Y_validation = np.array(Y_validation)
   # Y_validation2= np.array(Y_validation2)
   # Y_validation3= np.array(Y_validation3)
   # print(X_validation.shape)
   # print(Y_validation.shape)
   # print(Y_validation2.shape)
   # print(Y_validation3.shape)

    #print("------------------------------------------------------------------------------------")
    
    #load rnn
    path = Path(f"./rnns/RRNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.h5")
    
    if path.is_file():
        rnn=  load_model(f"./rnns/RRNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.h5")
        

    else:
        #early stopping
        early_stopping=EarlyStopping(patience=25, start_from_epoch=25, monitor="output1_mean_absolute_error",  restore_best_weights=True)
        
        #learning function
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        #Shape
        rnn_shape=(n_input, n_features)
        
        
        print(rnn_shape)
        #create model
        
        rnn = create_rnn(rnn_shape , 1, num_rnn)
        print(rnn.input_shape[1:]) 
        print(rnn.summary())
        

            
        hora_actual = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"-------------HORA DE COMIENZO DE TRAIN---> {hora_actual}-------------")
            
        history= rnn.fit(X_train, [Y_train, Y_train2, Y_train3], epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping, lr_scheduler])

        hora_actual = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"-------------HORA DE FINALIZACIÓN DE TRAIN---> {hora_actual}-------------")

        #Guardamos modelo e historial de entrenamiento
        rnn.save(f"./rnns/RRNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.h5")
            
        with open(f"./history/RRNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.json", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
        with open(f"./history/RRNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.json", "rb") as file_pi:
            load_history = pickle.load(file_pi)
        
        print(load_history)
        
        #plot train
        pd.DataFrame(history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,0.1)
        plt.show()
        plt.close()
    
    #Guardamos predicciones
    X_test = [] #Los 100 datos por cada predicción
    Y_test = [] #Aprendizaje supervisado
    Y_test2= []
    Y_test3= []
    aux=[]
    time_step = 100 #Bloques de 100 datos (100 anteriores por cada salida)
        
    print("---------------------------------------------TEST---------------------------------------")
    for i in range(time_step-1,len(test_data)): #Este buvcle está bien hecho, desde la posicion 99 hasta el final
        Y_test.append(test_data[i,22])
        Y_test2.append(test_data[i,23])
        Y_test3.append(test_data[i,24])
        aux=[]

        for j in range(time_step-1,-1,-1):
            aux.append(test_data[i-j,0:22]) #En la primera ejecucion son del 0-99, luego 1-100 y así hasta el final
            #print(i-j)
        X_test.append(aux) 


    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_test2= np.array(Y_test2)
    Y_test3= np.array(Y_test3)
    print(X_test.shape)
    print(Y_test.shape)
    print(Y_test2.shape)
    print(Y_test3.shape)
    


    print("------------------------------------------------------------------------------------")
        
        

            
    predictions = rnn.predict(X_test)
    print(np.array(predictions[0]).shape)
    
    
    real=Y_test
    prediction= predictions[0]
    
    real2=Y_test2
    prediction2= predictions[1]
    
    real3=Y_test3
    prediction3= predictions[2]
    
    print(real.shape)
    print(prediction.shape)
    
    
    #Guardamos salida
    predictions= np.array(predictions)
    predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
    pd.DataFrame(predictions).to_csv(f'./predictions/RNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.csv', index=False, header=True)
    
    #Comprobamos lo guardado
    #predictions = pd.read_csv(f'./predictions/RNN_{num_rnn}_DATASET_{num_data}_REP_{rep}.csv')
    #predictions= np.array(predictions)

    #Persistencias:
    
    persistence_error= mean_squared_error(test_data[n_input:,22], test_data[n_input:,21])
    mse = mean_squared_error(real, prediction)
    
    print("Persistence Error 1d: {:.4f}".format(persistence_error))
    print("MSE 1d: {:.4f}".format(mse))
    
    
    persistence_error2= mean_squared_error(test_data[n_input:,23], test_data[n_input:,21])
    mse2 = mean_squared_error(real2, prediction2)
    
    print("Persistence Error 5d: {:.4f}".format(persistence_error2))
    print("MSE 5d: {:.4f}".format(mse2))

    
    persistence_error3= mean_squared_error(test_data[n_input:,24], test_data[n_input:,21])
    mse3 = mean_squared_error(real3, prediction3)
    
    print("Persistence Error 10d: {:.4f}".format(persistence_error3))
    print("MSE 10d: {:.4f}".format(mse3))


    plt.plot(real, label='Precio de cierre real')
    plt.plot(prediction, label='Conjunto predicho')
    plt.xlabel('Día')
    plt.ylabel('Precio de cierre (USD)- Sigue sin desnormalizar')
    plt.title('Predicción del precio de las acciones: 1 dia')
    plt.legend()
    plt.show()
    plt.close()
    
    
    
    plt.plot(real2, label='Precio de cierre real')
    plt.plot(prediction2, label='Conjunto predicho')
    plt.xlabel('Día')
    plt.ylabel('Precio de cierre (USD)- Sigue sin desnormalizar')
    plt.title('Predicción del precio de las acciones: 5 dias')
    plt.legend()
    plt.show()
    plt.close()
    
    
    plt.plot(real3, label='Precio de cierre real')
    plt.plot(prediction3, label='Conjunto predicho')
    plt.xlabel('Día')
    plt.ylabel('Precio de cierre (USD)- Sigue sin desnormalizar')
    plt.title('Predicción del precio de las acciones: 10 dias')
    plt.legend()
    plt.show()
    plt.close()
    
    
    
    #A ver como entrena
    predictions_train = rnn.predict(X_train)
    
    plt.plot(Y_train, label='Precio de cierre real')
    plt.plot(predictions_train[0], label='Conjunto predicho')
    plt.xlabel('Día')
    plt.ylabel('Precio de cierre (USD)- Sigue sin desnormalizar')
    plt.title('TRAAAIN 1d')
    plt.legend()
    plt.show()
    plt.close()
    
    

    
    
    

def select_dataset(num_data):

    
    switch_dict_data = {
        #Big companies (<10.000M
        0: "AAPL", #Apple
        1: "GOOGL", #Google 
        2: "ITX.MC", #Inditex
        
        #Medium companies (1.000-10.000 M)
        3: "CAF", #CAF es una empresa española de fabricación de trenes
        4: "VIS.MC", # Visvofan, empresa española para bolsas de comida
        5: "ROVI.MC", #Empresa farmaceútica española
        
        #Small companies (> 1.000M)
        6: "GIGA", #Gigas Hosting, empresa española de soluciones hosting
        7: "ATRY.MC", #Atrys Health, empresa española de salud digital
        8: "ADZ.MC", #Adolfo dominguez
        
        #Criptomonedas
        9: "SOL-USD", #Solana
        10: "ETH-USD", #Etherium
        11: "BNB-USD" #Binance

    }

    
    if num_data in switch_dict_data:
        company = yf.Ticker(switch_dict_data[num_data])
        history = company.history(start="1900-01-01", interval="1d") #Pongo fecha antigua para que salgan todos los datos
        df = pd.DataFrame(history)
    
        # Convertir el índice en formato de fecha
        df.index = pd.to_datetime(df.index)
        
        
        #Añadimos indicadores (los 34 primeros no tendrán valor claro)
        
        df['SMA'] = ta.trend.sma_indicator(df['Close'], window=20)                                                  #1.-Promedio Móvil Simple (SMA) 
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)                                                  #2.-Promedio Móvil Exponencial (EMA)
        df['MACD'] = ta.trend.macd(df['Close'], 26,12)                                                              #3.- Media Móvil de la Convergencia/Divergencia (MACD)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)                                                         #4.-Índice de Fuerza Relativa (RSI)
        df['AO'] = ta.momentum.AwesomeOscillatorIndicator(df["High"], df["Low"], 5, 34).awesome_oscillator()        #5.- Oscilador Asombroso (AO)
        df['STOCH'] = ta.momentum.StochasticOscillator(df["High"], df["Low"],df["Close"], 20,5).stoch()             #6.- Oscilador Estocástico (STOCH)
        df['ADX'] = ta.trend.ADXIndicator(df["High"], df["Low"],df["Close"], 14).adx()                              #7.-Average Directional Movement Index (ADX)
        df['MFI'] = ta.volume.MFIIndicator(df["High"], df["Low"],df["Close"], df["Volume"], 14).money_flow_index()  #8.-Money Flow Index (MFI)
        df['NVI'] = ta.volume.NegativeVolumeIndexIndicator(df["Close"], df["Volume"]).negative_volume_index()       #9.-Negative Volume Index (NVI)
        df['PSAR'] = ta.trend.PSARIndicator(df["High"], df["Low"],df["Close"], 0.02, 0.2).psar()                    #10.-Parabolic Stop and Reverse (Parabolic SAR)
        #11.-Bandas de Bollinger   
        df['BB_UP'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_hband() 
        df['BB_MID'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_mavg()
        df['BB_LOW'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_lband() 
        df = df.drop(df.index[0:33]) #Eliminamos las 26 primeras muestras para no tener indicadores incompletos

        # Generar las columnas día, mes y año a partir del índice
        
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['WeekDay'] = df.index.weekday
        
        #Delete some columns
        
        df = df.drop(['Dividends'], axis=1)
        df = df.drop(['Stock Splits'], axis=1)
        
        #Put Low y Low shifted at final
        cols = df.columns.tolist()
        cols.append(cols.pop(cols.index('Low')))
        df = df[cols]
        
        
        #Generar columnas de salida
        horizonte=0 #Día siguiente
        df['Low_shifted_1'] = df['Low'].shift(periods=-(horizonte+1))
        
        horizonte=4 #Dentro de 5
        df['Low_shifted_5'] = df['Low'].shift(periods=-(horizonte+1))
        
        horizonte=9 #Dentro de 10 días
        df['Low_shifted_10'] = df['Low'].shift(periods=-(horizonte+1))
        df = df.drop(df.index[range(-1,-(2+horizonte),-1)])
        

        return df
    else:
        print("Dataset not defined.")   
    


    
#ALL THE MODELS
def create_rnn(input_shape, output_shape, num_rnn):

    
    switch_dict = {
        
        0: rnn_DENSE,
        1: rnn_GRU,
        2: rnn_LSTM,
        3: rnn_CONV,
        4: rnn_CONV_LSTM
        
   

    }
    
    if num_rnn in switch_dict:
        return switch_dict[num_rnn](input_shape, output_shape)
    else:
        print("RNN not defined.")    

    
#-----------------------------------------CUSTOM LEARNING RATE----------------------------------------------


from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    if epoch < 40:
        learning_rate = 0.001
    elif epoch<60:
        learning_rate = 0.0001
    else:
        learning_rate = 0.000001
    return learning_rate




        
# --------------------------------------MODELS----------------------------------------------------------

def rnn_DENSE(input_shape, output_shape):


    # Camino de entrada
    input_layer = Input(shape=input_shape)

    dense_layer_1 = Dense(units = 60, activation='relu')(input_layer)
    
    dense_layer_2 = Dense(units = 60, activation='relu')(dense_layer_1)
    
    dense_layer_3 = Dense(units = 60, activation='relu')(dense_layer_2)

    dropout_1 = Dropout(0.2)(dense_layer_3)
    
    
    #Camino 1 día--------------------------------------------------------------------------------------------------------------
    dense_layer_4_1 = Dense(units = 60, activation='relu')(dropout_1)

    
    dense_layer_5_1 = Dense(units = 60, activation='relu')(dense_layer_4_1)
 

    
    dropout_2_1 = Dropout(0.2)(dense_layer_5_1)
    flatten_1 = Flatten()(dropout_2_1)
    
    output1=Dense(units = 1, name='output1')(flatten_1)
 
    
    
    
    #Camino 5 días--------------------------------------------------------------------------------------------------------------
    dense_layer_4_2 = Dense(units = 60, activation='relu')(dropout_1)

    dense_layer_5_2 = Dense(units = 60, activation='relu')(dense_layer_4_2)


  
    dropout_2_2 = Dropout(0.2)(dense_layer_5_2)
    flatten_2 = Flatten()(dropout_2_2)
    
    output2=Dense(units = 1, name='output2')(flatten_2)

    #Camino 10 dias--------------------------------------------------------------------------------------------------------------
    dense_layer_4_3 = Dense(units = 60, activation='relu')(dropout_1)
    
    dense_layer_5_3 = Dense(units = 60, activation='relu')(dense_layer_4_3)

    

  
    dropout_2_3 = Dropout(0.2)(dense_layer_5_3)
    flatten_3 = Flatten()(dropout_2_3)
    
    output3=Dense(units = 1, name='output3')(flatten_3)

    
    #Modelo
    model = Model(inputs=input_layer, outputs=[output1, output2, output3])
    model.compile(optimizer="adam", loss={'output1': "mean_squared_error", 'output2': "mean_squared_error", 'output3': "mean_squared_error"}, metrics=["mean_absolute_error"])
    
    

    return model

def rnn_GRU(input_shape, output_shape):
    
    # Camino de entrada  
 
    input_layer = Input(shape=input_shape)
    
    gru_layer_1=GRU(units=60, return_sequences=True)(input_layer)
    
    gru_layer_2=GRU(units=60, return_sequences=True)(gru_layer_1)
    
    gru_layer_3=GRU(units=60, return_sequences=True)(gru_layer_2)
    
    dropout_1 = Dropout(0.2)(gru_layer_3)
    
    
    
    #Camino 1 día--------------------------------------------------------------------------------------------------------------
    gru_layer_4_1=GRU(units=60, return_sequences=True)(dropout_1)
    
    gru_layer_5_1=GRU(units=60, return_sequences=True)(gru_layer_4_1)
    
    dropout_2_1 = Dropout(0.2)(gru_layer_5_1)
    
    output1_layerFlat = Flatten()(dropout_2_1)
    
    output1=Dense(units = 1, name='output1')(output1_layerFlat)

    
    
    #Camino 5 días--------------------------------------------------------------------------------------------------------------
    gru_layer_4_2=GRU(units=60, return_sequences=True)(dropout_1)
    
    gru_layer_5_2=GRU(units=60, return_sequences=True)(gru_layer_4_2)
    
    dropout_2_2 = Dropout(0.2)(gru_layer_5_2)
    
    output2_layerFlat = Flatten()(dropout_2_2)
    
    
    output2=Dense(units = 1, name='output2')(output2_layerFlat)

    
    
    #Camino 10 dias--------------------------------------------------------------------------------------------------------------
    gru_layer_4_3=GRU(units=60, return_sequences=True)(dropout_1)
    
    gru_layer_5_3=GRU(units=60, return_sequences=True)(gru_layer_4_3)
    
    dropout_2_3 = Dropout(0.2)(gru_layer_5_3)
    
    output3_layerFlat = Flatten()(dropout_2_3)

    
    output3=Dense(units = 1, name='output3')(output3_layerFlat)
 
    
    #Modelo
    model = Model(inputs=input_layer, outputs=[output1, output2, output3])
    model.compile(optimizer="adam", loss={'output1': "mean_squared_error", 'output2': "mean_squared_error", 'output3': "mean_squared_error"}, metrics=["mean_absolute_error"])
    
    return model

        #POSIBLES CONFIGURACIONES:
            #Número de neuronas


def rnn_LSTM(input_shape, output_shape):


    # Camino de entrada
    input_layer = Input(shape=input_shape)

    lstm_layer_1 = LSTM(units = 60, return_sequences = True)(input_layer)
     
    lstm_layer_2 = LSTM(units = 60, return_sequences = True)(lstm_layer_1)

    lstm_layer_3 = LSTM(units = 60, return_sequences = True)(lstm_layer_2)
    
    dropout_1 = Dropout(0.2)(lstm_layer_3)
    
    
    #Camino 1 día--------------------------------------------------------------------------------------------------------------
    lstm_layer_4_1 = LSTM(units = 60, return_sequences = True)(dropout_1)

    lstm_layer_5_1 = LSTM(units = 60, return_sequences = True)(lstm_layer_4_1)
    
    dropout_2_1 = Dropout(0.2)(lstm_layer_5_1)
    output1_layerFlat = Flatten()(dropout_2_1)

    output1=Dense(units = 1, name='output1')(output1_layerFlat)
 
    
    
    #Camino 5 días--------------------------------------------------------------------------------------------------------------
    lstm_layer_4_2 = LSTM(units = 60, return_sequences = True)(dropout_1)
    
    lstm_layer_5_2 = LSTM(units = 60, return_sequences = True)(lstm_layer_4_2)
    
    dropout_2_2 = Dropout(0.2)(lstm_layer_5_2)
    output2_layerFlat = Flatten()(dropout_2_2)
    

    output2=Dense(units = 1, name='output2')(output2_layerFlat)

    
    
    #Camino 10 dias --------------------------------------------------------------------------------------------------------------
    lstm_layer_4_3 = LSTM(units = 60, return_sequences = True)(dropout_1)
    
    lstm_layer_5_3 = LSTM(units = 60, return_sequences = True)(lstm_layer_4_3)
    
    dropout_2_3 = Dropout(0.2)(lstm_layer_5_3)
    output3_layerFlat = Flatten()(dropout_2_3)
    

    output3=Dense(units = 1, name='output3')(output3_layerFlat)


    #Modelo
    model = Model(inputs=input_layer, outputs=[output1, output2, output3])
    model.compile(optimizer="adam", loss={'output1': "mean_squared_error", 'output2': "mean_squared_error", 'output3': "mean_squared_error"}, metrics=["mean_absolute_error"])
    
    return model




def rnn_CONV(input_shape, output_shape):


    # Camino de entrada
    input_layer = Input(shape=input_shape)

    conv_layer_1 = Conv1D(60, 3,    activation='relu')(input_layer)
    
    conv_layer_2 = Conv1D(60, 3,   activation='relu')(conv_layer_1)
    
    conv_layer_3 = Conv1D(60, 3,    activation='relu')(conv_layer_2)

    dropout_1 = Dropout(0.2)(conv_layer_3)
    
    
    #Camino 1 día--------------------------------------------------------------------------------------------------------------
    conv_layer_4_1 = Conv1D(60, 3,   activation='relu')(dropout_1)

    
    conv_layer_5_1 = Conv1D(60, 3,   activation='relu')(conv_layer_4_1)
 

    
    dropout_2_1 = Dropout(0.2)(conv_layer_5_1)
    flatten_1 = Flatten()(dropout_2_1)
    
    output1=Dense(units = 1, name='output1')(flatten_1)
 
    
    
    
    #Camino 5 días--------------------------------------------------------------------------------------------------------------
    conv_layer_4_2 = Conv1D(60, 3,   activation='relu')(dropout_1)

    conv_layer_5_2 = Conv1D(60, 3,   activation='relu')(conv_layer_4_2)


  
    dropout_2_2 = Dropout(0.2)(conv_layer_5_2)
    flatten_2 = Flatten()(dropout_2_2)
    
    output2=Dense(units = 1, name='output2')(flatten_2)

    #Camino 10 dias--------------------------------------------------------------------------------------------------------------
    conv_layer_4_3 = Conv1D(60, 3,   activation='relu')(dropout_1)
    
    conv_layer_5_3 = Conv1D(60, 3,   activation='relu')(conv_layer_4_3)

    

  
    dropout_2_3 = Dropout(0.2)(conv_layer_5_3)
    flatten_3 = Flatten()(dropout_2_3)
    
    output3=Dense(units = 1, name='output3')(flatten_3)

    
    #Modelo
    model = Model(inputs=input_layer, outputs=[output1, output2, output3])
    model.compile(optimizer="adam", loss={'output1': "mean_squared_error", 'output2': "mean_squared_error", 'output3': "mean_squared_error"}, metrics=["mean_absolute_error"])
    
    

    return model

    
            



def rnn_CONV_LSTM(input_shape, output_shape):


    # Camino de entrada
    input_layer = Input(shape=input_shape)

    conv_layer_1 = Conv1D(60, 3,   activation='relu')(input_layer)
    
    lstm_layer_2 = LSTM(units = 60, return_sequences = True)(conv_layer_1)
    
    conv_layer_3 = Conv1D(60, 3,   activation='relu')(lstm_layer_2)
    
    dropout_1 = Dropout(0.2)(conv_layer_3)
    
    
    #Camino 1 día--------------------------------------------------------------------------------------------------------------
    lstm_layer_4_1 = LSTM(units = 60, return_sequences = True)(dropout_1)
    
    conv_layer_5_1 = Conv1D(60, 3,   activation='relu')(lstm_layer_4_1)
    
    dropout_5_1 = Dropout(0.2)(conv_layer_5_1)
    output1_layerFlat = Flatten()(dropout_5_1)

    output1=Dense(units = 1, name='output1')(output1_layerFlat)

    
    
    #Camino 5 días--------------------------------------------------------------------------------------------------------------
    lstm_layer_4_2 = LSTM(units = 60, return_sequences = True)(dropout_1)

    
    conv_layer_5_2 = Conv1D(60, 3,   activation='relu')(lstm_layer_4_2)
    
    dropout_5_2 = Dropout(0.2)(conv_layer_5_2)
    output2_layerFlat = Flatten()(dropout_5_2)
    
    output2=Dense(units = 1, name='output2')(output2_layerFlat)
 
    
    #Camino 10 dias--------------------------------------------------------------------------------------------------------------
    lstm_layer_4_3 = LSTM(units = 60, return_sequences = True)(dropout_1)

    
    conv_layer_5_3 = Conv1D(60, 3,   activation='relu')(lstm_layer_4_3)
    
    dropout_5_3 = Dropout(0.2)(conv_layer_5_3)
    output3_layerFlat = Flatten()(dropout_5_3)
    

    output3=Dense(units = 1, name='output3')(output3_layerFlat)

    


    #Modelo
    model = Model(inputs=input_layer, outputs=[output1, output2, output3])
    model.compile(optimizer="adam", loss={'output1': "mean_squared_error", 'output2': "mean_squared_error", 'output3': "mean_squared_error"}, metrics=["mean_absolute_error"])
    
    return model
