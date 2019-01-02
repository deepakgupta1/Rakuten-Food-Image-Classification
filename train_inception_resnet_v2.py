import keras
import pandas as pd
import numpy as np

CROP_SIZE = 299
NUM_CLASSES = 43

train_data_dir = '../../rakuten/train/'
validation_data_dir = '../../rakuten/validation/'
nb_train_samples = 471436
nb_validation_samples = 83220

def random_crop(x):    
    w, h = x.shape[0], x.shape[1]
    rangew = (w - CROP_SIZE)
    rangeh = (h - CROP_SIZE)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped =  x[offsetw:offsetw+CROP_SIZE, offseth:offseth+CROP_SIZE]
        
    return cropped
	
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input

image_gen = ImageDataGenerator(rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    channel_shift_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)


val_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

BATCH_SIZE = 16
SEED = 1

train_gen = image_gen.flow_from_directory(train_data_dir, batch_size=BATCH_SIZE, seed=SEED, target_size=(CROP_SIZE, CROP_SIZE), shuffle=True, class_mode = "categorical")
valid_gen = val_image_gen.flow_from_directory(validation_data_dir, batch_size=BATCH_SIZE, seed=SEED, target_size=(CROP_SIZE, CROP_SIZE), shuffle=True, class_mode = "categorical")

from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D 

def get_model():
    keras.backend.clear_session()
    model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(CROP_SIZE, CROP_SIZE,3), pooling='avg') 
    y = Dense(64, activation='relu')(model.layers[-1].output)
    y = Dropout(0.3)(y)
    y = Dense(64, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(NUM_CLASSES, activation='softmax')(y)
    model = Model(inputs=model.input, outputs=y)
    for l in model.layers[:-1]:
        l.trainable = False
    
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model
	
filepath = '../../rakuten/model-inception-resnet-v2-retrain-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, min_lr=0.000001, verbose=1, min_delta=0.001)

callbacks_list = [checkpointer, early, reduce_lr]

from keras.models import load_model

keras.backend.clear_session()
model = get_model()

model.fit_generator(train_gen,
                      steps_per_epoch=nb_train_samples//BATCH_SIZE, 
                      epochs=2, 
                      verbose=1, 
                      callbacks=callbacks_list, 
                      validation_data=valid_gen, 
                      validation_steps=nb_validation_samples//BATCH_SIZE,
                      max_queue_size=20, 
                      shuffle=True,
                      workers=4)

for l in model.layers[:-1]:
    l.trainable = True
    
sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['acc'])
model.summary()
model.fit_generator(train_gen,
                      steps_per_epoch=nb_train_samples//BATCH_SIZE,
                      epochs=200, 
                      verbose=1, 
                      callbacks=callbacks_list, 
                      validation_data=valid_gen, 
                      validation_steps=nb_validation_samples//BATCH_SIZE,
                      max_queue_size=20, 
                      shuffle=True,
                      workers=4)
					  
model.save('../../rakuten/rakuten_inception_resnet_v2_retrain_model.h5')					  
