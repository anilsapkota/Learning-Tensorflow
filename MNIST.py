import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

#import data 
data = keras.datasets.fashion_mnist

#split the data into train and test

(train_images,train_labels),(test_images,test_labels) = data.load_data()

#different class labels of the MNIST fashion dataset

class_names = ['Tshirt','Trousers','Pullover','Dress','Coat',
                'Sandal','Shirt','Sneaker','Bag','Ankle Boot']

train_images = train_images/255.0  #normalizing the data 
test_images = test_images/255.0

#print(train_images[7]) #actual way the computer sees the image
#display the image of the dataset
#plt.imshow(train_images[7])
#plt.show()

#Creating a model 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #we are passing flattened images as the input 
    keras.layers.Dense(128,activation="relu"), #for hidden layers we took like 50% as the number of neurons 
    keras.layers.Dense(10,activation="softmax") #softmax is the probability distribution 
])

#compile the model 
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

#train our model 
model.fit(train_images,train_labels,epochs=5)

#we want to evlauate how our model does on test set to check the accuracy 
# test_loss, test_acc = model.evaluate(test_images,test_labels)

# print("Tested Acc:" , test_acc)

#use the model to predict the value 

prediction = model.predict(test_images)

# print(np.argmax(prediction[0]))  #displays the class number of the prediction of the test image 
# print(class_names[np.argmax(prediction[0])])  #displays the corresponding name of class of the prediction of the test image

#for loop for prediction and results with the display of the image  

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("Actual:" + class_names[test_labels[i]])
    plt.title("prediction: "+class_names[np.argmax(prediction[i])])
    plt.show()
