import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt 


#load the data
data = keras.datasets.imdb

#we are only take those words that are 10000 most frequent while loading 
(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words=10000)

# print(train_data[0]) #prints the integer encoded words 

word_index = data.get_word_index() #gives the tuples of string and word in them 

#k stands for key and v stands for value 
word_index = {k:(v+3)for k , v in word_index.items()}

word_index["<PAD>"]=0  #making same length movie list by padding 
word_index["<START>"]= 1 # 
word_index["<UNK>"] =2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key)for (key,value)in word_index.items()])

#normalize the data to 250 words 

train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=250)

# print(len(test_data),len(test_data))
def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

# print(decode_review(test_data[0])) #This prints the actual text 

# print(len(test_data[0]),len(test_data[1])) #it tells that each input is of different lengths 

#define the model 

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16)) #make the word vectors similiar based on grouping
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation = "relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

x_val = train_data[:10000]
x_train =train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]


# fit model 
fitModel = model.fit(x_train,y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)

results = model.evaluate(test_data,test_labels)

print(results)
