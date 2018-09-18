# -*- coding: utf-8 -*-

# Convolutional Neural Network


# Importing the Keras libraries and packages
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import glob
import os





# Loss class
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'



training_set_path = "dataset/training_set"
test_set_path = "dataset/test_set"


history = LossHistory()

def save_model(model):
    model.save('models/my_model.h5')
    
    
def cargar_modelo():
    model = load_model('models/my_model.h5')
    return model

def save_weights(model):
    model.save_weights('models/model_weights.h5')
    
def cargar_weights(model):
    model.load_weights('models/model_weights.h5')
    return model
    


def convolution_network_model():
    # Initialising the CNN
    classifier = Sequential()
    # Convolution
    # Creating feature map, with 3,3 filter
    # 64 cause we are using a easier format better use 256 256
    classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
    # Pooling
    # Reduce the size of the featuring map, most of the time 2, 2 for not losing information
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Flattening
    classifier.add(Flatten())    
    # Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  # cross entropy better for classification
    return classifier




#https://keras.io/preprocessing/image/
def train_model(epoch=50):
    classifier = convolution_network_model()
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(training_set_path,
                                                     target_size = (128, 128),
                                                     batch_size = 32,
                                                     shuffle=True,
                                                     class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory(test_set_path,
                                                target_size = (128, 128),
                                                batch_size = 32,
                                                shuffle=True,
                                                class_mode = 'binary')

    
    
    classifier.fit_generator(training_set,
                             steps_per_epoch = None,
                             epochs = epoch,
                             validation_data = test_set,
                             validation_steps = None,
                             callbacks=[history])

    print("The model class indices are:", training_set.class_indices)
    return classifier




import numpy as np
from keras.preprocessing import image
    
def predict_image(model, ruta):   
    test_image = image.load_img(ruta, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    return result



from keras import applications
from keras.models import Model



def vgg16_train_model(epoch=25, load=False):
    if load:
        model = load_model('models/vgg16.h5')
        model.load_weights('models/vgg16_weights.h5')
    else:
        base_model = applications.VGG16(weights='imagenet', input_shape = (128, 128, 3), include_top=False) #None for random init
        
        
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))
        top_model.summary()
    
        # add the model on top of the convolutional base
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        # Set the first mode layers not trainable
        for layer in model.layers[:15]:
            layer.trainable = False
            
        #model.summary()
        
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory(training_set_path,
                                                         target_size = (128, 128),
                                                         batch_size = 32,
                                                         shuffle=True,
                                                         class_mode = 'binary')
        
        test_set = test_datagen.flow_from_directory(test_set_path,
                                                    target_size = (128, 128),
                                                    batch_size = 32,
                                                    shuffle=True,
                                                    class_mode = 'binary')
    
        
        
        model.fit_generator(training_set,
                                 steps_per_epoch = None,
                                 epochs = epoch,
                                 validation_data = test_set,
                                 validation_steps = None,
                                 callbacks=[history])
        model.save('models/vgg16.h5')
        model.save_weights('models/vgg16_weights.h5')
        loss_history_path = os.path.join("loss_history_vgg.log")
        myFile = open(loss_history_path, 'w+')
        myFile.write(history.losses)
        myFile.close()
        print("The model class indices are:", training_set.class_indices)
    return model


