#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


x_train = train_datagen.flow_from_directory(r'C:\Users\HP\Desktop\data_set\TRAIN_SET\TRAIN_SET',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test = test_datagen.flow_from_directory(r'C:\Users\HP\Desktop\data_set\TEST_SET-20221101T044129Z-001\TEST_SET',target_size=(64,64),batch_size=32,class_mode='categorical')


# In[8]:


x_train.class_indices


# In[9]:


model = Sequential()


# In[10]:


model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation ="relu"))


# In[11]:


model.add(MaxPooling2D(pool_size =(2,2)))


# In[12]:


model.add(Flatten())


# In[14]:


model.add(Dense(units = 128 ,kernel_initializer ="uniform" , activation = "relu"))


# In[15]:


model.add(Dense(units = 5 ,kernel_initializer ="uniform" , activation = "softmax"))


# In[18]:


model.compile(optimizer = "adam", loss="categorical_crossentropy",metrics = ["accuracy"])


# In[19]:


model.fit_generator(x_train,steps_per_epoch=47,epochs=10,validation_data=x_test ,validation_steps =20)


# In[20]:


model.save("fruit.h5")


# In[21]:









