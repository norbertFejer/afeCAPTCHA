# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


AGGREGATE_BLOCK_NUM = 1


class Classifier_FCN:
    
    def __init__(self, output_directory, input_shape, nb_classes, nb_filters=128, verbose=False, build=True):
        self.output_directory = output_directory
        
        if build == True:
            self.nb_filters = nb_filters
            self.model = self.build_model(input_shape, nb_classes)

            if(verbose == True):
                self.model.summary()
            
            self.verbose = verbose
            self.model.save_weights(self.output_directory + '/fcn_model_init_nb_f' + str(self.nb_filters) + '.hdf5')
            
        return
        
    
    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        
        conv1 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
        
        conv2 = keras.layers.Conv1D(filters=self.nb_filters*2, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        
        conv3 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
        
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])
            
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)
            
        file_path = self.output_directory + '/best_fcn_model_nb_f' + str(self.nb_filters) + '.hdf5'
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]
        
        return model
        
    
    def fit(self, trainX, trainy):
        
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 16
        nb_epochs = 10

        mini_batch_size = int(min(trainX.shape[0]/10, batch_size))

        start_time = time.time()

        x_train, x_valid, y_train, y_valid = train_test_split(trainX, trainy, test_size=0.25, shuffle=False)

        hist = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
            batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, callbacks=self.callbacks)
		
        duration = time.time() - start_time
        print("Model learning finished in: ", str(duration) + "s")
    
        self.model.save(self.output_directory + 'last_fcn_model.hdf5')
        return hist
		
