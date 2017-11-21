[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# README #
## Overview ##

   The objective of the Follow Me project, designed by Udacity, was to locate and follow a target, the target being a person in red, using a quadracopter in a virtual city. A full understanding of neural networks and deep learning was pertanint for succeeding in this project. A Fully Convolutional Neural Network (FCN) was used to solve this perception classification problem. First, individual camera frames from the drone's front facing camera were analyzed and used to train the FCN. Once trained, the FCN could classify each pixel from each frame, this is known as Semanic Segmentation. Techniques utilized in this project can be applicable to the problems seen in robotic image classification and depth inference problems.   

### Hardward and Software Used ###  

Keras, high level deep learning API for tensor flow

Amazons Web Services for GPU


## Architecture ##

This section will review the FCN  used in this project. I will outline why the FCN was chosen for this application, and explain the layers to this network. 

![FCN Diagram](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/FCN_Diagram.PNG)

Above is a diagram of the FCN Model created for this project, below is the code for the FCN. 

```python
def fcn_model(inputs, num_classes):

    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    conv_layer1=encoder_block(inputs,filters=32,strides=1)
    conv_layer2=encoder_block(conv_layer1,filters=64,strides=1)
    conv_layer3=encoder_block(conv_layer2,filters=128,strides=1)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
        concat_layer=conv2d_batchnorm(conv_layer3,filters=256,kernel_size=3,strides=1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    deco_layer1=decoder_block(concat_layer,conv_layer2,filters=128)
    deco_layer2=decoder_block(deco_layer1,conv_layer1,filters=64)
    output_layer=decoder_block(deco_layer2,inputs,filters=32)

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(output_layer)
```

### FCN Overview ###

The difference between a Fully Connnected Neural Network and an FCN is that the FCN replaces the fully connected layers with a 1x1 convolutional layer. By having this layer of convolution, spatial information is preserved throughout the entire network. Such spatial information is important for classify the structure of an object. For a Fully Connected Network, each neuron is connected to every neuron in the previous layer, each of these connections contain their own weight.Such unique weights being held in memory create expensive computations. FCNs use local connections, so each neuron is connected to a few nearby neurons in the previous layer. As well the same weights are used for each neuron. The benefit to the FCN is that the data can be intepretated as spatial and extracted features are spatially local, meaning that they can occur at any input location. Since an FCN can interpret an object in any input location, this makes it very useful for things like analzying images, i.e. where is the target in the photo. For this reason an FCN is perfect neural network for this application.  

### Encoder ###

The encoder is the first section of the FCN, the encoder extracts features that will later be used by the decoder. The encoder is the portion which reduces to a 1x1 convolutional layer. The FCN created in this project includes separable convolution layers using the previous layer and filter depth. Seperable convolutions is a technique used that reduces the number of parameters needed thus increases efficiency, and reducing overfitting. Initially I started the project with only two convolution layers, I added a third layer on the 3rd trail in attempt to increase the IOU score (defined later).   

### 1X1 Convolution ###

Output of convolutional layer is a 4D tensor feedin this through a fully connected layer flattens it into a 2D tensor , spatial information is lost. This is avoided through 1x1 convolutions. 1x1 convolution layer reduces dimensionalit of the layer. A benefit of the 1x1 convolutional layer is that you can test images of any size.  
batch normalization - normalizes each layer's inputs by using mean and variabce of the values in the current mini batch. Advantages of this are train fater and faster convergence. Allows high learning rate . Simplifies creation fo deeper networks and provides a bit of regulation

### Decoder ### 
For Decoders, reverses convolutions
helps unsampling previous layer to desired resolution or dimension
creates unsampleing layer using bilinear unsampling , has a layer of concatenation, 
Bilinear Unsampling is the weighted average of nearest pixels located diagonally to a given pixel
skip connections are used here, skip connections retain information. How they work is information from previous encoding layers or input skips adjacent layers and into the decoder or output layer. This creates more precise segmentation decisions as less information is lost.
layer concatenation is a way to carry ouyt skip connections. Concateting two layers the upsampling layer and a layer with more spatial information. allows for more flexability as depth of input layers nee not match


## Results ##
### Hyper Parameters ###
Below was the initial settings to the hyper parameters:
```python
learning_rate =0.001
batch_size =120
num_epochs = 10
steps_per_epoch = 200
validation_steps = 20
workers = 2
```
These were selected from the previous lab completed. Below is a table outlining the different trails, changes in parameters and their respected results. Results were scored as IOU, the intersection set over union set, it gives an idea of how well the classification handles ever single pixel.

| Trail         | Changes       | Result (IOU)|
| ------------- |:-------------:| -----:|
| 1    | Initial setup, 2 Convolution Layers|0.38 |
| 2    | Batch size=200, Validation Steps=50      |0.40 |
| 3    | Batch size=16, Validation Steps=200   |0.35 |
| 4    | 3 Convolution Layers |0.35 |
| 5    | Learning rate = 0.003 |0.32 |
| 6    | Batch size=32, Epochs=15 |0.40 |

Final settings : 
```python 
learning_rate =0.003
batch_size =32
num_epochs = 15
steps_per_epoch = 500
validation_steps = 200
workers = 2
```

![Training Curves](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/Training_Curves.PNG)
![Segmentation](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/Segmentation.PNG)
![IOU](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/Score.PNG)

## Future Enhancements ##


