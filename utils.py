##################
#######INFO#######
##################
"""
FILE: utils.py
MAIN FILE: bioinformatic
PYTHON: 3.8.3
AUTHOR: Sebastian Janampa & Cristina Mallqui
CREATE DATE:
"""


################# 
####LIBRARIES####
#################
import numpy as np
np.random.seed(1)# For reproducibility
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow import keras
from tensorflow.keras import activations, layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
tf.random.set_seed(1234)# For reproducibility


##########################################
##############IMPORT DATASET##############
##########################################
def load_data(file):
    lista=[]
    records= list(open(file, "r"))
    records=records[1:]
    for seq in records:
        elements=seq.split(",")
        level=elements[-1].split("\n")
        classe=level[0]
        lista.append(classe)

    lista=set(lista)
    classes=list(lista)
    X=[]
    Y=[]
    for seq in records:
        elements=seq.split(",")
        X.append(elements[1:-1])
        level=elements[-1].split("\n")
        classe=level[0]
        Y.append(classe)
    X=np.array(X,dtype=float)
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    Y=np.array(Y,dtype=int)
    data_max= np.amax(X)
    X = X/data_max
    return X,Y,len(classes),len(X[0])


##########################################
##############DEEP LEARNING###############
##########################################
class myRepModel(keras.Model):
    def __init__(self, nb_classes):
        super().__init__(name='PaperModel')
        tf.random.set_seed(1234)# For reproducibility
        self.conv1 = Convolution1D(filters=5, kernel_size=5, padding='valid',
                                  name='Conv1')
        self.max1 = MaxPooling1D(pool_size=2, padding='valid',
                                name='MaxPool1')
        self.conv2 = Convolution1D(filters=10, kernel_size=5, padding='valid',
                                  name='Conv2')
        self.max2 = MaxPooling1D(pool_size=2, padding='valid',
                                name='MaxPool2')
        
        self.fc1 = Dense(units=500, name='Dense1')
        self.dp = Dropout(0.5, name='Dropout')
        self.fc2 = Dense(nb_classes, name='Dense2')
        
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = tf.nn.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.max2(x)
        x = Flatten(name='Flatten')(x)
        x = self.fc1(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = activations.softmax(x)
        return x


# Basic convolutional block (CONV-RELU-BATCH)
class BlockConv1Dv1(layers.Layer):
    def __init__(self, out_channels, kernel_size=5, stride=1, padding='same', name=None):
        super().__init__(name=name)
        self.conv = Convolution1D(filters=out_channels, 
                                  kernel_size=kernel_size,
                                  strides=stride,
                                  padding=padding, 
                                  kernel_initializer='he_normal')
        # Relu activation function
        self.bn = layers.BatchNormalization()
    
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = activations.relu(x)
        x = self.bn(x, training=training)
        return x
    

# This model is based on VGG
class VGG(keras.Model):
    def __init__(self, model, nb_classes):
        self.parameters = {
        '11': [1, 1, 2, 2, 2],
        '13': [2, 2, 2, 2, 2],
        '16': [2, 2, 3, 3, 3],
        '19': [2, 2, 4, 4, 4]
    }
        super().__init__(name='VGG'+model)
        tf.random.set_seed(1234)# For reproducibility
        layers = self.parameters[model]
        self.blocks = []
        out_channels = 5
        # Convolutional layers
        self.blocks.append(BlockConv1Dv1(out_channels=5, stride=2, padding='valid'))
        self.blocks.append(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        for num_layers in layers:
            for _ in range(num_layers):
                self.blocks.append(BlockConv1Dv1(out_channels))
            self.blocks.append(MaxPooling1D(pool_size=2, strides=2, padding='same'))
            if out_channels < 256:
                out_channels *= 2
        # Multi-connected layers (After each dense layer we applied a RELU activation)
        self.dense1 = Dense(1024, kernel_initializer='he_normal')
        self.dense2 = Dense(1024, kernel_initializer='he_normal')
        self.outputs = Dense(nb_classes)
    def call(self, input_tensor, training=False):
        x = input_tensor
        for block in self.blocks:
            x = block(x, training=training)
        x = Flatten()(x)
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x)
        x = activations.relu(x)
        x = self.outputs(x)
        x = activations.softmax(x)
        return x
    
# Basic convolutional block (BATCH-RELU-CONV)
class BlockConv1Dv2(layers.Layer):
    def __init__(self, out_channels, kernel_size=5, stride=1, padding='same', name=None):
        super().__init__(name=name)
        self.bn = layers.BatchNormalization()
        # Relu activation function
        self.conv = Convolution1D(filters=out_channels, 
                                  kernel_size=kernel_size,
                                  strides=stride,
                                  padding=padding, 
                                  kernel_initializer='he_normal')
    
    def call(self, input_tensor, training=False):
        x = self.bn(input_tensor, training=training)
        x = activations.relu(x)
        x = self.conv(x)
        return x

# ResNet block (5x1CONV-5x1CONV)
class ResBlockv1(layers.Layer):
    def __init__(self, filters, downsample, name=None):
        super().__init__(name=name)
        self.block1 = BlockConv1Dv2(filters, stride=2 if downsample else 1)
        self.block2 = BlockConv1Dv2(filters)
        self.downsample = downsample
        if downsample:
            self.identity_map = BlockConv1Dv2(filters, kernel_size=1, stride=2)
    
    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        if self.downsample:
            identity = self.identity_map(input_tensor, training=training)
        else:
            identity = input_tensor
        x += identity
        return x

# ResNet block (1x1CONV-5x1CONV-1x1CONV)
class ResBlockv2(layers.Layer):
    def __init__(self, filters, downsample, name=None):
        super().__init__(name=name)
        self.block1 = BlockConv1Dv2(filters, kernel_size=1)
        self.block2 = BlockConv1Dv2(filters, stride=2 if downsample else 1)
        self.block3 = BlockConv1Dv2(filters*4, kernel_size=1)
        self.downsample = downsample
        if downsample:
            self.identity_map = BlockConv1Dv2(filters*4, kernel_size=1, stride=2)
    
    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        if self.downsample:
            identity = self.identity_map(input_tensor, training=training)
        else:
            identity = input_tensor
        x += identity
        return x
# This model is based on ResNet
class ResNet(keras.Model):
    def __init__(self, model, nb_classes):
        super().__init__(name=model)
        parameters = {
            'ResNet18':{'block_type': ResBlockv1,
                        'layers': [2, 2, 2, 2]},
            'ResNet34':{'block_type': ResBlockv1,
                        'layers': [3, 4, 6, 3]},
            'ResNet50':{'block_type': ResBlockv2,
                        'layers': [3, 4, 6, 3]}
        }
        block_type, num_filters = parameters[model].values()
        tf.random.set_seed(1234)# For reproducibility
        self.blocks = []
        out_channels = 5
        prev_channels = 5 if block_type == ResBlockv1 else 0
        # Convolutional layers
        self.blocks.append(BlockConv1Dv1(out_channels=5, stride=2, padding='valid'))
        self.blocks.append(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        for num in num_filters:
            for _ in range(num):
                downsample = out_channels != prev_channels
                block_type(filters=out_channels, downsample=downsample)
                prev_channels = out_channels
            prev_channels *= 2
        # Multi-connected layers
        self.outputs = Dense(nb_classes)
        
    def call(self, input_tensor, training=False):
        x = input_tensor
        for block in self.blocks:
            x = block(x, training=training)
        x = GlobalAveragePooling1D()(x)
        x = self.outputs(x)
        x = activations.softmax(x)
        return x
        
        
def myModel(modeltype, nb_classes, model=None):
    if model is None:
        my_model = modeltype(nb_classes)
    else:
        my_model = modeltype(model, nb_classes)
    my_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return my_model

# Training
def train_and_evaluate_model (model, datatr, labelstr, datate, labelste,nb_classes, num_epochs, callbacks=False, batchsize=256):
    datatr = datatr.reshape(datatr.shape + (1,))
    datate = datate.reshape(datate.shape + (1,))
    labelstr = keras.utils.to_categorical(labelstr, nb_classes)
    labelste_bin = keras.utils.to_categorical(labelste, nb_classes)
    if callbacks:
        class myCallback(keras.callbacks.Callback):
            """Display the metrics of the model every 20 epochs"""
            def __init__(self, num_epochs):
                self.epochs = num_epochs

            def on_epoch_end(self, epoch, logs=None):
                if (epoch%40==39) or (epoch==0) or (epoch==99):
                    print("    Epoch %04i/%04i:"%(epoch+1, self.epochs), end=' ')
                    for key, value in logs.items():
                        print('%s: %.3f'%(key, value), end=' - ')
                    print()
        callback = myCallback(num_epochs)
        model_history = model.fit(x=datatr, y=labelstr, 
                                  validation_data=(datate, labelste_bin),
                                  epochs=num_epochs, 
                                  batch_size=batchsize, 
                                  verbose=0,
                                  callbacks=callback
                                 )
        return model_history
        
    else:
        callback = None
        model.fit(x=datatr, y=labelstr, 
                  validation_data=(datate, labelste_bin),
                  epochs=num_epochs, 
                  batch_size=batchsize, 
                  verbose=0,
                  callbacks=callback
                 )
        tr_scores = model.evaluate(datatr,labelstr,verbose=0)
        scores = model.evaluate(datate, labelste_bin,verbose=0)
        print('Training_acc:  %.3f --- Testing_acc: %.3f'%(tr_scores[1], scores[1]))
        return scores[1]

# Plotting
def plot_results(history):
    fig, axs = plt.subplots(1,1)
    fig.set_figheight(10)
    fig.set_figwidth(13)
    for name, hist in history.items():
        if name[-5]=='O':
            if name[4]=='S':
                color = '#BA82F5'
                lab_name = 'Order-SG'
            else:
                color = '#7748A8'
                lab_name = 'Order-AMP'
        elif name[-5]=='C':
            if name[4]=='S':
                color = '#DB5546'
                lab_name = 'Class-SG'
            else:
                color = '#F58D82'
                lab_name = 'Class-AMP'
        elif name[-5]=='F':
            if name[4]=='S':
                color = '#F5CB64'
                lab_name = 'Family-SG'
            else:
                color = '#EBD28C'
                lab_name = 'Family-AMP'
        elif name[-5]=='G':
            if name[4]=='S':
                color = '#A6EB73'
                lab_name = 'Genus-SG'
            else:
                color = '#62EB00'
                lab_name = 'Genus-AMP'
        axs.plot(hist.history['accuracy'], color = color, label=lab_name, linewidth=1.5, linestyle='-')
        axs.plot(hist.history['val_accuracy'], color = color, linewidth=1.5, linestyle='--')
        axs.grid(True)
        axs.legend(loc='upper left')
        axs.set_xlabel('Epoch', fontsize=15)
        axs.set_title('Accuracy', fontsize=18, fontweight='bold')