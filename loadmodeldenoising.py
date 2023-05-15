from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# In[]: load model
model = keras.models.load_model('mnistaedenoising.h5')


# In[]: load data
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
test_images = x_test[9000:]


factor = 0.39
test_images = test_images.astype('float32') / 255.0
test_images = np.reshape(test_images,(test_images.shape[0],28,28,1))
test_noisy_images = test_images + factor * np.random.normal(loc = 0.0,scale = 1.0,size = test_images.shape)


# In[]: predict
plt.figure(figsize = (18,18))
for i in range(10,19):  
    if(i == 45):
        plt.title('Denoised Images', fontsize = 25, color = 'Blue') 
    
    plt.subplot(9,9,i)
    res=model.predict(test_noisy_images[i].reshape(1,28,28,1))
    plt.imshow(res.reshape(1,28,28)[0], cmap = plt.cm.binary) 
plt.show()