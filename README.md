# Rakuten-Food-Image-Classification

### Problem.
The challenge is to develop an algorithm that would classify a given recipe food image into one of the pre-defined categories with high precision and recall. A labeled dataset is provided for supervised training and mapping to different food categories. It's a top-3 class predictios challenge, means that for every food image, the top 3 classes to which it might belong is to be predicted.

### Data.
The Rakuten recipe food image data is distributed across 43 categories with around 0.79 million images. Each image category is prepared according to their Recipe, the time taken to prepare it and it's ingredients etc. 

### Approach.
* Supervised deep learning approach.
* Fine tune multiple models, pretrained on imagenet dataset.
* Added a couple of fully connected layers with dropout.
* First fine tune these extra layers with high learning rate of 0.001 and stochastic gradient descent optimizer for a couple of epochs(~3).
* Then fine tune the whole network with lower learning rate of 0.0001 and sgd.

### Models used.
* InceptionV3.
* ResNet50.
* Densenet201.
* InceptionResNetV2.

### Preprocessing and augmentation:
* Random cropping from center of image.
* Data augmentation using random rotation, horizontal/vertical flip, zoom, width/height/channel shift.

### Train/validation split:
* Randomly split the whole dataset to 85% training and 15% validation stratified over the 43 classes.

### Ensembling:
* The probabilities of all the trained models are averaged for final result and top 3 classes are picked from this averaged probability.

### Libraries/packages:
* Keras
* Pandas
* Numpy	
