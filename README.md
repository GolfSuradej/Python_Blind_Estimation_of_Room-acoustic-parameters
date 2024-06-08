Install Python and Environment setup 

Conda
```
conda create -n room-acoustics python=3.7.15
conda activate room-acoustics
```
Venv
```
pyenv install 3.7.15
python -m venv
source venv/bin/activate
```
pyenv for Windows: https://pypi.org/project/pyenv-win/


Models Noisy #The models were trained under noisy reverberant environments

checkpoints: https://gofile.me/72Wjs/YbNrD9AJw

A model for estimating speech transmission index (STI) Only

https://gofile.me/72Wjs/yFVEDvDT8



Datasets

Room Impulse Responses (RIRs), SMILE dataset: https://gofile.me/72Wjs/J4zDC6cqu

Test Sound (convoluted with SMILE RIRs): https://gofile.me/72Wjs/hNL4HyhvG

ACE dataset: https://gofile.me/72Wjs/KE6u8zagt


Runing--Example

```
import os
import math
import numpy as np
from tensorflow import keras
import librosa
import scipy
from scipy.io import wavfile
from scipy.signal import butter, hilbert, filtfilt
import acoustics
from acoustics.signal import bandpass
from acoustics.bands import (octave_low, octave_high)

samplerate = 48000
Fs = samplerate
fc = 30 # cutoff frequency of the speech envelope  at 30 Hz
N = 3   #filter order
w_L = 2*fc/Fs
num, den = butter(N,w_L, 'low') #lowpas filter tf
bands = acoustics.bands.octave(125,8000)

path = './models_noisy/'
def loadModels():
    global model_1
    global model_2
    global model_3
    global model_4
    global model_5
    global model_6
    global model_7
    model_1 = keras.models.load_model(os.path.join(path, 'estT60snr_ch1.h5'))
    model_2 = keras.models.load_model(os.path.join(path, 'estT60snr_ch2.h5'))
    model_3 = keras.models.load_model(os.path.join(path, 'estT60snr_ch3.h5'))
    model_4 = keras.models.load_model(os.path.join(path, 'estT60snr_ch4.h5'))
    model_5 = keras.models.load_model(os.path.join(path, 'estT60snr_ch5.h5'))
    model_6 = keras.models.load_model(os.path.join(path, 'estT60snr_ch6.h5'))
    model_7 = keras.models.load_model(os.path.join(path, 'estT60snr_ch7.h5'))


def estMTFs(signal, fs):
        L = fs*5
        if len(signal) > fs * 5:
            signal = signal[0:fs * 5]
        else:
            signal = np.pad(signal, (0, L - len(signal)), mode='constant')
    
        bands = acoustics.bands.octave(125,8000)

        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])

        filtered_signal = np.zeros([7, 220500])
        for band in range(bands.size):
            filtered_signal[band] = bandpass(signal, low[band], high[band], fs, order=8)

        fc = 30  # cutoff frequency of the speech envelope  at 30 Hz
        N = 3  # filter order
        w_L = 2 * fc / fs
        # lowpas filter tf
        num, den = butter(N, w_L, 'low')
        PEs = np.zeros([7, 2000])
        for k in range(7):
            ey = filtfilt(num, den, np.abs(hilbert(filtered_signal[k] ** 2)))
            PEs[k] = scipy.signal.resample(ey, 2000)

        x_test = PEs
        estT60snr = np.zeros([7, 2])

        x_test = x_test.astype('float32')
        x_test_1 = x_test[0, :]
        x_test_2 = x_test[1, :]
        x_test_3 = x_test[2, :]
        x_test_4 = x_test[3, :]
        x_test_5 = x_test[4, :]
        x_test_6 = x_test[5, :]
        x_test_7 = x_test[6, :]

        PE_1 = x_test_1.reshape((1, 2000, 1))
        PE_2 = x_test_2.reshape((1, 2000, 1))
        PE_3 = x_test_3.reshape((1, 2000, 1))
        PE_4 = x_test_4.reshape((1, 2000, 1))
        PE_5 = x_test_5.reshape((1, 2000, 1))
        PE_6 = x_test_6.reshape((1, 2000, 1))
        PE_7 = x_test_7.reshape((1, 2000, 1))
        
        estT60snr[0, :] = model_1.predict(PE_1)
        estT60snr[1, :] = model_2.predict(PE_2)
        estT60snr[2, :] = model_3.predict(PE_3)
        estT60snr[3, :] = model_4.predict(PE_4)
        estT60snr[4, :] = model_5.predict(PE_5)
        estT60snr[5, :] = model_6.predict(PE_6)
        estT60snr[6, :] = model_7.predict(PE_7)
                
        for k in range(7):
             snr = estT60snr[k, 1]
             T60 = estT60snr[k, 0]
             print("Band %d T60: %1.2f, SNR: %1.1f" %(k, T60, snr))
        return estT60snr

loadModels()
wavfile = 'SoundCheck/t20301_J_FAFSA304.wav'
signal, Fs = librosa.load(wavfile, sr = 44100)

estT60SNRs = estMTFs(signal, Fs)
estMTF = np.zeros([7,30])
for k in range(7):
    for f in range(30):
        estMTF[k,f] = (1/(math.sqrt(1+((2*math.pi*f*estT60SNRs[k, 0])/13.8)**2)))*(1/(1+10**(-estT60SNRs[k, 1]/10)))
        #print(estMTF[k,f])

```

