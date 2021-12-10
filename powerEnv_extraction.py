# get Power envelope feature
bands = acoustics.bands.octave(125,8000)

low = third_low(bands[0], bands[-1])
high = third_high(bands[0], bands[-1])

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
