import numpy as np
import librosa
import IPython.display as ipd
import scipy
from scipy.signal import butter, hilbert, filtfilt

import keras
from keras.models import load_model
import tensorflow as tf

from scipy.signal import butter, hilbert, filtfilt
import acoustics
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
import matplotlib.pyplot as plt


import tensorflow as tf



def MTFbasedCNNs_estRIRparams(PEs):  
    x_test = PEs
    estT60snr = np.zeros([7, 2])
    
    x_test = x_test.astype('float32') 
    x_test_1 = x_test [0,:]
    x_test_2 = x_test [1,:]
    x_test_3 = x_test [2,:]
    x_test_4 = x_test [3,:]
    x_test_5 = x_test [4,:]
    x_test_6 = x_test [5,:]
    x_test_7 = x_test [6,:]
    
    TAE_1 = x_test_1.reshape((1,300,1))
    TAE_2 = x_test_2.reshape((1,300,1))
    TAE_3 = x_test_3.reshape((1,300,1))
    TAE_4 = x_test_4.reshape((1,300,1))
    TAE_5 = x_test_5.reshape((1,300,1))
    TAE_6 = x_test_6.reshape((1,300,1))
    TAE_7 = x_test_7.reshape((1,300,1))
    
    estT60snr[0,:] = model_1.predict(TAE_1)
    estT60snr[1,:] = model_2.predict(TAE_2)
    estT60snr[2,:] = model_3.predict(TAE_3)
    estT60snr[3,:] = model_4.predict(TAE_4)
    estT60snr[4,:] = model_5.predict(TAE_5)
    estT60snr[5,:] = model_6.predict(TAE_6)
    estT60snr[6,:] = model_7.predict(TAE_7)
        
    return estT60snr


#load model keras.models.load_model
model_1 = load_model('estT60snr_ch1.h5')
model_2 = load_model('estT60snr_ch2.h5')
model_3 = keras.models.load_model('estT60snr_ch3.h5')
model_4 = keras.models.load_model('estT60snr_ch4.h5')
model_5 = keras.models.load_model('estT60snr_ch5.h5')
model_6 = keras.models.load_model('estT60snr_ch6.h5')
model_7 = keras.models.load_model('estT60snr_ch7.h5')


cd AppliedAcoustics_Lab/1_SMILE43speech
signal, fs = librosa.load('408_J_FAFSA304.wav',44100)



# get Power envelope feature
bands = acoustics.bands.octave(125,8000)

low = octave_low(bands[0], bands[-1]) 
high = octave_high(bands[0], bands[-1]) 
#low = third_low(bands[0], bands[-1])
#high = third_high(bands[0], bands[-1])

filtered_signal = np.zeros([7,220500])
for band in range(bands.size):
    filtered_signal[band] = bandpass(signal, low[band], high[band], fs, order=8)
    
#power envelop extraction
fc = 30 # cutoff frequency of the speech envelope  at 30 Hz
N = 3   #filter order
w_L = 2*fc/fs
#lowpas filter tf
num, den = butter(N,w_L, 'low')

PEs = np.zeros([7,300])

for k in range(7):
    ey = filtfilt(num,den,np.abs(hilbert(filtered_signal[k])))
    ey =scipy.signal.resample(ey,300)
    PEs[k] = ey**2




T60snr = MTFbasedCNNs_estRIRparams(PEs)
print(T60snr)
