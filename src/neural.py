import random
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano

class NNmaker():

    def __init__(self):
        self.PopDNA = []
        # DNA Straind {NumN : number of neurons, NumL : number of layers, LI: layer inits,   }

    def reproduce_CNN(self,DNA,inputs, classes):

        ''' Makes a multi-layer-perceptron neural network depending on a DNA strand'''

        model = Sequential()
        num_neurons_in_layer = 500
        num_inputs = X_train.shape[1]
        num_classes = y_train_ohe.shape[1]


        model.add(Dense(input_dim=(num_inputs * 2),
                         output_dim=num_neurons_in_layer,
                         init='random_uniform',
                         activation='relu'))
         # only 12 neurons - keep softmax at last layer
        sgd = SGD(lr=0.0044, decay=1e-7, momentum=.9) # using stochastic gradient descent (keep)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] ) # (keep)
        return model

    def makeRanMLP(self, inputs, classes):
        ''' Function to make a random multi-layer-perceptron '''
        DNA = {}
        actList = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']
        model = Sequential()
        NumN = random.randint(0,500)
        DNA['NumN'] = NumN
        num_inputs = inputs
        num_classes = classes
        print(num_inputs)
        print(num_classes)
         # only 12 neurons in this layer!
        layers = random.randint(0,5)
        LM = []
        LA = []
        LK = 'random_uniform'
        DNA['NumL'] = layers
        act = random.choice(actList)

        img_rows, img_cols = 28, 28
        nb_filters = 12
        pool_size = (2, 2)
        kernel_size = (4, 4)
        input_shape = (1, img_rows, img_cols)
        model = Sequential()
        model.add(Conv2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
        model.add(Activation('tanh'))
        model.add(Flatten())
        for r in range(layers):
            lm = random.randint(0,5)
            act = random.choice(actList)
            LM.append(lm)
            LA.append(act)
            if r != 1:
                model.add(Dense(input_dim=(num_inputs),
                             units=NumN,
                             kernel_initializer= LK,
                             activation= act))

        model.add(Dense(input_dim=NumN,
                         units=num_classes,
                         kernel_initializer='uniform',
                         activation='softmax'))
        DNA['LM'] = LM
        DNA['LA'] = LA
        DNA['LI'] = LK

        LR = random.uniform(0.1, 0.0001)
        DNA['LR'] = LR
        sgd = SGD(lr=LR, decay=1e-7, momentum=.9) # using stochastic gradient descent (keep)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"] )
        self.PopDNA.append(DNA)
        return model, DNA


    def makerandom_CNN(self, X_train, X_test, classes):
        '''Function to make a random Convolution Neural Net'''
        #dont change
        img_rows, img_cols = 300, 300
        nb_classes = classes
        #-------------
        DNA={}
        batch_size = randint(1,5000)
        DNA['BS']=batch_size
        epochs = randint(1,10)
        DNA['epochs']=epochs
        nb_filters = randint(10,20)
        DNA['NumFileter']=nb_filters
        x = randint(1,6)
        pool_size = (x, x)
        DNA['PoolSize']=pool_size
        x = randint(2,20)
        kernel_size = (4, 4)
        DNA['KS']=kernel_size


        #dont change
        model = Sequential()
        input_shape = (img_rows, img_cols,3)
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        #----------------------

        #options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

        c2d_layers = randint(1,2)
        DNA['C2DLayers']= c2d_layers
        c2d_Activ = []
        activ = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']
        for x in range(c2d_layers):
            act = random.choice(activ)
            model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                                    padding='valid',
                                    input_shape=input_shape))
            model.add(Activation(act))
            c2d_Activ.append(act)
        DNA['c2d_Activ']= c2d_Activ

        #dont change
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.5))
        model.add(Flatten())
        #-------------------

        Dense_Activ = []
        Dense_Nur = []
        DenseLayers = randint(1,2)
        DNA['DenseLayers']= DenseLayers
        for x in range(DenseLayers):
            NN = randint(1,10)
            act = random.choice(activ)
            model.add(Dense(NN))
            model.add(Activation(act))
            Dense_Activ.append(act)
            Dense_Nur.append(NN)
        DNA['Dense_Activ']= Dense_Activ
        DNA['Dense_Nur']= Dense_Nur

        drop = random.uniform(0.1, 0.7)
        model.add(Dropout(drop))
        DNA['Drop'] = drop
        #dont change ----------------------
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        #----------------------------

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model, DNA


    def reporduce_CNN(self, X_train, X_test, classes, DNA):
        '''Function to make a CNN from a DNA strand'''
        #dont change
        #{'BS': 4840, 'NumFileter': 14, 'PoolSize': (4, 4), 'KS': (4, 4), 'C2DLayers': 1, 'c2d_Activ': ['relu'], 'DenseLayers': 8, 'Dense_Activ': ['sigmoid', 'sigmoid', 'softsign', 'linear', 'softsign', 'tanh', 'linear', 'softplus'], 'Dense_Nur': [26, 304, 467, 776, 817, 359, 369, 641], 'Drop': 0.3098848045887589}#
            #dont change
            img_rows, img_cols = 300, 300
            nb_classes = classes
            #-------------
            batch_size = DNA['BS']
            nb_filters = DNA['NumFileter']
            pool_size = DNA['PoolSize']
            kernel_size = DNA['KS']



            #dont change
            model = Sequential()
            input_shape = (img_rows, img_cols,3)
            # if K.image_dim_ordering() == 'th':
            #     X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            #     X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            #     input_shape = (3, img_rows, img_cols)
            # else:
            #     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            #     input_shape = (img_rows, img_cols, 3)

            #----------------------

            #options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

            c2d_layers = DNA['C2DLayers']
            c2d_Activ =  DNA['c2d_Activ']
            if  c2d_layers != len(c2d_Activ):
                while len(c2d_Activ) < c2d_layers :
                    c2d_Activ.append(c2d_Activ[0])
                    DNA['c2d_Activ'] = c2d_Activ
                while len(c2d_Activ) > c2d_layers :
                    c2d_Activ.pop()
                    DNA['c2d_Activ'] = c2d_Activ
                    print(c2d_Activ)


            for x in range(c2d_layers):
                act = c2d_Activ[x]
                model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                                        padding='valid',
                                        input_shape=input_shape))
                model.add(Activation(act))


            #dont change
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(Dropout(0.5))
            model.add(Flatten())
            #-------------------

            Dense_Activ = DNA['Dense_Activ']
            Dense_Nur = DNA['Dense_Nur']
            DenseLayers = DNA['DenseLayers']
            DNA['DenseLayers']= DenseLayers

            if DenseLayers != len(Dense_Activ):
                print('Increase: ' + str(Dense_Activ))
                while len(Dense_Activ) < DenseLayers :
                    Dense_Activ.append(Dense_Activ[0])
                    DNA['Dense_Activ'] = Dense_Activ
                    print(Dense_Activ)
                while len(Dense_Activ) > DenseLayers :
                    Dense_Activ.pop()
                    DNA['Dense_Activ'] = Dense_Activ
                    print(Dense_Activ)
            if DenseLayers != len(Dense_Nur):
                while len(Dense_Nur) < DenseLayers :
                    Dense_Nur.append(Dense_Nur[0])
                    DNA['Dense_Nur'] = Dense_Nur
                    print(Dense_Nur)
                while len(Dense_Nur) > DenseLayers :
                    Dense_Nur.pop()
                    DNA['Dense_Nur'] = Dense_Nur
                    print(Dense_Nur)


            for x in range(DenseLayers):
                model.add(Dense(Dense_Nur[x]))
                model.add(Activation(Dense_Activ[x]))

            model.add(Dropout(DNA['Drop']))
            #dont change ----------------------
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))
            #----------------------------

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            return model
