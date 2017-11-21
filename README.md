[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# README#
## Overview ##

   The objective of the Follow Me project, designed by Udacity, was to locate and follow a target, the target being a person in red, using a quadracopter in a virtual city. A full understanding of neural networks and deep learning was pertanint for succeeding in this project. A Fully Convolutional Neural Network (FCN) was used to solve this perception classification problem. First, individual camera frames from the drone's front facing camera were analyzed and used to train the FCN. Once trained, the FCN could classify each pixel from each frame, this is known as Semanic Segmentation. Techniques utilized in this project can be applicable to the problems seen in robotic image classification and depth inference problems.   

### Software used ###  
Keras, high level deep learning API for tensor flow
AWS

## Architecture ##

![FCN Diagram](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/FCN_Diagram.PNG)

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

FCNs are unqiue to Deep Neural Networks at they replace the fully connected layers with a 1x1 convolutional layer. By having this layer of convolution they preserve the spatial information throughout the entire network. Such spatial information is important for classify the structure of an object. 
By replacing the fully connected layers with a 1x1 convolutional layer a series of unsampleing through transposed convolutional layers is used to get to the output layer. This series of layers is known as the Decoder layer. Below


### Input ###

### Encoder ###

Creates a separable convolutional layer using an input layer and size of filters 
encoder extracts features that will later be used by the decoder. The encoder layers are the pretrained model. 
seperable convolutions is a technique used that reduces the number of parameters needed thus increases efficiency

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

### Output  ###
Intersection of union, gives and idea of how well the classification handles every single pixels

## Results ##
### Hyper Parameters ###
![Training Curves](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/Training_Curves.PNG)
![Segmentation](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/Segmentation.PNG)
![IOU](https://github.com/GlennPatrickMurphy/Follow_Me/blob/master/docs/Score.PNG)

## Future Enhancements ##


