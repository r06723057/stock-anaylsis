import numpy as np
import pandas as pd

data = pd.read_csv('C:\\data5\\newpp.csv')
#C:\\data5\\newpp.csv
#/home/u/b03201003/newpp.csv
data=data.iloc[:,1:]

data=data.values

for i in range(1,data.shape[1]-22):
    for j in range(0,data[:,0].size-1):
        data[j][i]=((data[j+1][i])-data[j][i])/data[j][i]

for i in range(data.shape[1]-22,data.shape[1]-15):
    for j in range(0,data[:,0].size-1):
        data[j][i]=((data[j+1][i])-data[j][i])

data=data[:-1,:]

c=np.zeros(data.shape[1]-13)
d=np.zeros(data.shape[1]-13)
for i in range(0,data.shape[1]-13):
    c[i]=np.std(data[:,i])
    d[i]=np.mean(data[:,i])

for i in range(0,data.shape[1]-13):
    for j in range(0,data[:,0].size):
        data[j][i]=((data[j][i])-d[i])/c[i]



TIME_STEPS = 50
dat = []
for i in range(data.shape[0] - TIME_STEPS ):
    dat.append(data[i: i + TIME_STEPS ,:])

reshaped_data = np.array(dat).astype('float64')
reshaped_data=reshaped_data

x = reshaped_data[:-1,:,:]
y = reshaped_data[1:,-1,1:5 ]

x1,x2,x3,x4,x5,x6,x7,x8=np.array_split(x,8,axis=0)
x=np.vstack((x1,x2,x3,x4,x5,x6,x7))

y1,y2,y3,y4,y5,y6,y7,y8=np.array_split(y,8,axis=0)
y=np.vstack((y1,y2,y3,y4,y5,y6,y7))

print(x.shape)
print(y.shape)

import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation,Dropout
from keras.optimizers import Adam, SGD
import keras
from keras import initializations
from keras.layers.normalization import BatchNormalization

BATCH_START = 0
BATCH_SIZE=32

INPUT_SIZE = 83
OUTPUT_SIZE = 4
CELL_SIZE = 500


model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    input_dim=INPUT_SIZE, input_length=TIME_STEPS,       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    #return_sequences=True,      # True: output at all steps. False: output as last step.
    init=keras.initializers.Orthogonal(gain=1.0, seed=None),
    #stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))


#model.add(LSTM(CELL_SIZE, init=keras.initializers.Orthogonal(gain=1.0, seed=None)))

model.add(Dropout(0.5))
#model.add(LSTM(CELL_SIZE, init=keras.initializers.Orthogonal(gain=1.0, seed=None)))

#model.add(LSTM(CELL_SIZE))
# add output layer
#model.add(Dropout(0.5))
model.add(Dense(OUTPUT_SIZE))

LR = 1e-3
adam = Adam(lr=LR,clipnorm=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam,
              loss='mse',)

model.fit(x, y,
    batch_size=64, epochs=10,validation_data=(x8, y8)
        )

from keras.optimizers import Adam, SGD
LR = 1e-1
sgd=SGD(LR,clipnorm=1e-6,  decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd,
              loss='mse',)

model.fit(x, y,
    batch_size=4, epochs=10,validation_data=(x8, y8)
        )

model.save('/home/u/b03201003/data/stock.h5')
 

predict=model.predict(x8[0][np.newaxis,:,:])
for i in range(1,x8.shape[0]):
    predict=np.concatenate((predict,model.predict(x8[i][np.newaxis,:,:])),axis=0)

for i in range(0,predict.shape[1]):
    for j in range(0,predict.shape[0]):
        predict[j][i]=(predict[j][i]+d[i])*c[i]

for i in range(0,y8.shape[1]):
    for j in range(0,y8.shape[0]):
        y8[j][i]=(y8[j][i]+d[i])*c[i]


print(predict)
print(y8)

evaluate( x, y )


x8.reshape([462,83])
y8.reshape([462,4])

predict=model.predict(x8)
import matplotlib.pyplot as plt
p=range(0,)
plt.plot(p, pre, label="pre", color="red", linewidth=2)
plt.plot(p, datadiv, "b--", label="y")
