import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import scipy.io.wavfile as wav
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

def show_waveform(fname, offset):
    rate, data = wav.read(fname)
    print(rate)
    print(len(data))
    plt.figure()
    plt.plot(data[offset*rate:int(offset+3.5)*rate])
    plt.show()
    return

girl = './data/girl.wav'
shubh = './data/shubh.wav'
baba = './data/swatantra.wav'

#show_waveform(girl, 0)
#show_waveform(shubh, 0)

def get_spectra_bands(fname, start, stop, window):
    rate, data = wav.read(fname)
    bands = []
    for i in np.arange(start, stop, window):
        a = data[int(i * rate):int((i + .5) * rate)]
        b = [(ele / 2 ** 8.0) * 2 - 1 for ele in a]
        c = fft(b)
        d = int(len(c) / 2)
        f = abs(c[:(d - 1)])
        bands.append(f[0:1000])
    nbands = np.array(bands)
    nbands **= .5
    return nbands

gBands = get_spectra_bands(girl,0,5, .25)     #y=0
sBands = get_spectra_bands(shubh,0,5, .25)    #y=1
bBands = get_spectra_bands(baba,0,5, .25)
rows,cols = gBands.shape
data = np.zeros((3*rows,cols+1),dtype='float')
data[0:rows,0:cols] = gBands
data[rows:2*rows,0:cols] = sBands
data[2*rows:,0:cols]=bBands
data[rows:,-1]=1
data[2*rows:,-1]=2

np.random.shuffle(data)

X = data[:,0:-1]
Y = data[:,-1]
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=.1)

# x_check = X_test[10]
# y_check = y_test[10]

#-------------------------------------------------------Training-------------------------------------------------
clf = LogisticRegression()
clf.fit(X_train,y_train)

#----------------------------------------Checking the accuracy of our model--------------------------------------
accuracy = clf.score(X_test,y_test)
print(accuracy*100)

x_check = bBands[3,:]
#print('Check data:',x_check)
print('Label :swatantra')
temp =int(clf.predict(x_check))
if temp==0:
    print('Output Label: Laundiya')
elif temp==1:
    print('Output Label: Shubham')
else:
    print('Swatantra label')



x_check = sBands[8,:]
#print('Check data:',x_check)
print('Label :shubham')
temp =int(clf.predict(x_check))
if temp==0:
    print('Output Label: Laundiya')
elif temp==1:
    print('Output Label: Shubham')
else:
    print('Swatantra label')

#
# print(data)
# print(gBands.shape)
# print("-------------------")
# print(sBands.shape)


