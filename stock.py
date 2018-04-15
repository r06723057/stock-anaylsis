mport numpy as np
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

xf=x.reshape(-1,x.shape[1],x.shape[2],1)
x8f=x8.reshape(-1,x.shape[1],x.shape[2],1)
y1,y2,y3,y4,y5,y6,y7,y8=np.array_split(y,8,axis=0)
y=np.vstack((y1,y2,y3,y4,y5,y6,y7))

y=y[:,3]
y8=y8[:,3]
print(x.shape)
print(y.shape)

import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation,Dropout, Multiply,Conv2D,Flatten
from keras.optimizers import Adam, SGD
import keras
from keras import initializations
from keras.layers.normalization import BatchNormalization

BATCH_START = 0
BATCH_SIZE=32

INPUT_SIZE = 83
OUTPUT_SIZE = 1
CELL_SIZE = 256
from keras.layers.core import*
from keras.models import Sequential

from keras.models import Model
model1 = Sequential()
# build a LSTM RNN
model1.add(LSTM(
    input_dim=INPUT_SIZE,input_length=TIME_STEPS,       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    units=CELL_SIZE,
    return_sequences=True,      # True: output at all steps. False: output as last step.
    kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),activity_regularizer=regularizers.l1(0.001)
    #stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
model1.summary()

#The LSTM  model -  output_shape = (batch, step, hidden)

#The weight model  - actual output shape  = (batch, step)
# after reshape : output_shape = (batch, step,  hidden)
model2 = Sequential()

model2.add(Conv2D(filters = 1, kernel_size = (1,INPUT_SIZE),padding = 'valid', 
                  input_shape = (TIME_STEPS,INPUT_SIZE,1),activity_regularizer=regularizers.l1(0.01)))
model2.add(Activation('softmax'))

model2.summary()
model2.add(Flatten())
#model2.add(Dense(input_shape=(INPUT_SIZE,TIME_STEPS), output_dim=TIME_STEPS))
#model2.add(Activation('softmax')) # Learn a probability distribution over each  step.
#Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
model2.add(RepeatVector(CELL_SIZE))
model2.add(Permute((2, 1)))

from keras import backend as K
#The final model which gives the weighted sum:
model = Sequential()

multi=keras.layers.Multiply()([model1.output, model2.output])

#model.add(Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]

multi=Lambda(function=lambda x: K.sum(x, axis=1), 
                   output_shape=lambda shape: (shape[0],) + shape[2:])(multi)

multi=Dropout(0.5)(multi)
multi=Dense(1)(multi)

#model.add(TimeDistributedMerge('sum')) # Sum the weighted elements.

model = Model([model1.input,model2.input], multi)
#model.add(LSTM(CELL_SIZE, init=keras.initializers.Orthogonal(gain=1.0, seed=None)))
model.summary()

#model.add(LSTM(CELL_SIZE, init=keras.initializers.Orthogonal(gain=1.0, seed=None)))

#model.add(LSTM(CELL_SIZE))
# add output layerhs
#model.add(Dropout(0.5))


LR = 1e-3
adam = Adam(lr=LR,clipnorm=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam,
              loss='mse',)

model.fit([x,xf], y,
    batch_size=64, epochs=3,validation_data=([x8,x8f], y8)
        )
for i in range(len(model.get_config()['layers'])):
    print(i)
    for key in model.get_config()['layers'][i]:
        print(key)
        print(model.get_config()['layers'][i][key])
i=0
for layer in model1.layers:
    g=layer.get_config()
    h=layer.get_weights()
    print (g)
    print (h)
    i=i+1
    if i==2:
        break


from keras.optimizers import Adam, SGD
LR = 1e-1
sgd=SGD(LR,clipnorm=1e-6,  decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd,
              loss='mse',)

model.fit(x, y,
    batch_size=4, epochs=10,validation_data=(x8, y8)
        )

model.save('/home/u/b03201003/data/stock.h5')
 
'''
predict=model.predict(x_p[0][np.newaxis,:,:])
for i in range(1,x_p.shape[0]):
    predict=np.concatenate((predict,model.predict(x_p[i][np.newaxis,:,:])),axis=0)

for i in range(0,predict.shape[1]):
    for j in range(0,predict.shape[0]):
        predict[j][i]=(predict[j][i]+d[i])*c[i]

for i in range(0,y_p.shape[1]):
    for j in range(0,y_p.shape[0]):
        y_p[j][i]=(y_p[j][i]+d[i])*c[i]

data = pd.read_csv('C:\\data5\\newpp.csv')
#C:\\data5\\newpp.csv
#/home/u/b03201003/newpp.csv
data=data.iloc[:,1:]

data=data.values

import matplotlib.pyplot as plt
p=range(0,len(predict[:,0]))
plt.plot(p, predict[:,0], label="pre", color="red", linewidth=0.1)
plt.plot(p, y_p[:,0], "b--", label="y",linewidth=0.1)
plt.show()
'''
