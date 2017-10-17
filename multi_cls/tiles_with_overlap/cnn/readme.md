# CNN model
### Preprocessing
* One sample splitted into 9 main tiles 64x64. Plus 4 overlapping
  tiles on the edges of main tiles. Then each tile resized to 32x32.
* Dataset of all tiles standardized (centered + scaled).
### Augmentation
90 degrees, [0.5, 2] zoom, reflect
### Architecture
![Architecture](architecture.png)
### Model summary
Layer (type)                | Output Shape           |   Param #   
----------------------------|------------------------|-------------
conv2d_1 (Conv2D)           | (None, 30, 30, 32)     |   320       
conv2d_2 (Conv2D)           | (None, 28, 28, 32)     |   9248      
max_pooling2d_1 (MaxPooling2| (None, 14, 14, 32)     |   0         
dropout_1 (Dropout)         | (None, 14, 14, 32)     |   0         
conv2d_3 (Conv2D)           | (None, 12, 12, 64)     |   18496     
batch_normalization_1 (Batch| (None, 12, 12, 64)     |   256       
conv2d_4 (Conv2D)           | (None, 10, 10, 64)     |   36928     
max_pooling2d_2 (MaxPooling2| (None, 5, 5, 64)       |   0         
flatten_1 (Flatten)         | (None, 1600)           |   0         
dense_1 (Dense)             | (None, 256)            |   409856    
dropout_2 (Dropout)         | (None, 256)            |   0         
dense_2 (Dense)             | (None, 111)            |   28527     
----------------------------|------------------------|-------------
Total params: 503,631  
Trainable params: 503,503  
Non-trainable params: 128  
### Results
![Loss plot](loss.png)
