import numpy as np
import pandas as pd
import os

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image


#datagen = ImageDataGenerator(zoom_range = 0.2, width_shift_range = 0.1, rotation_range = 30, rescale=1./255)
datagen = ImageDataGenerator(rescale=1./255)

print('started...')

# load the model we saved
model1 = load_model('../../rakuten/rakuten_inception_v3_retrain_model.h5')
model2 = load_model('../../rakuten/rakuten_resnet50_retrain_model.h5')
model3 = load_model('../../rakuten/rakuten_densenet201_retrain_model.h5')
model4 = load_model('../../rakuten/rakuten_inception_resnet_v2_retrain_model.h5')
#model5 = load_model('../../rakuten/model-rsn50-ft-use-10-1.75-0.5287.hdf5')
print('models loaded!')

img_width, img_height = 299, 299
test_data_dir = '../../rakuten/test/'

categories = []
img_names = []
images = []

count = 0

generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(299, 299),
        batch_size=256,
        seed=27,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels
		
img_names = generator.filenames

probs1 = model1.predict_generator(generator, verbose=1)
probs2 = model2.predict_generator(generator, verbose=1)
probs3 = model3.predict_generator(generator, verbose=1)
probs4 = model4.predict_generator(generator, verbose=1)

probs = [(p+q+r+s)/4.0 for p,q,r,s in zip(probs1, probs2, probs3, probs4)]

for prob in probs:
	top_values_index = sorted(range(len(prob)), key=lambda i: prob[i])[-3:]
	categories.append(','.join([str(x) for x in top_values_index]))
	
# model learned labels to TShirt category mapping
category_mapping = {'0':'0', '1':'1', '2':'10', '3':'11', '4':'12', '5':'13', '6':'14', '7':'15', '8':'16', '9':'17', '10':'18', '11':'19', '12':'2', '13':'20', '14':'21', '15':'22', '16':'23', '17':'24',
                   '18':'25', '19':'26', '20':'27', '21':'28', '22':'29', '23':'3', '24':'30', '25':'31', '26':'32', '27':'33', '28':'34', '29':'35', '30':'36', '31':'37', '32':'38', '33':'39', '34':'4',
                   '35':'40', '36':'41', '37':'42', '38':'5', '39':'6', '40':'7', '41':'8', '42':'9'}


df = pd.DataFrame(columns=['Image-Id', 'Category'])
df['Image-Id'] = img_names
df['Category'] = categories

df['Category'] = df['Category'].apply(lambda x: ','.join([category_mapping[str(y)] for y in x.split(',')]))
df['Image-Id'] = df['Image-Id'].apply(lambda x: x[13:])

df.to_csv('../../rakuten/submit.csv', index=False)