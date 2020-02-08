import keras 
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D,SpatialDropout2D
from keras.layers import LeakyReLU,Concatenate,Activation,Cropping2D,Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, Input,BatchNormalization,UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from typing import Tuple,Union,List

def Conv3x3DB(filters: Tuple[int,int],
              kernel_size: Tuple[int,int],
              padding: str,
              activation: str,
              use_batchnorm: bool = True,**kwargs):
    def wrapper(input_tensor): 
        x = Conv2D(filters=filters,kernel_size=kernel_size,
                        padding=padding,activation=activation,**kwargs)(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(axis= 3 if keras.backend.image_data_format()=='channels_last' else 1)(x)
        x = Conv2D(filters=filters,kernel_size=kernel_size,
                        padding=padding,activation=activation,**kwargs)(x)
        return x
    return wrapper 
    
def MaxPoolBlock(pool_size: Tuple[int,int]):
    def wrapper(input_tensor): 
        x = MaxPooling2D(pool_size=pool_size)(input_tensor)
        return x 
    return wrapper
    
def UpSampleBlock(filters: Tuple[int,int],
                  kernel_size: Tuple[int,int],
                  padding: str,
                  upsize: Tuple[int,int],
                  upsample: bool = None):
    def wrapper(up_conv,conc_conv): 
        if upsample: 
            x = UpSampling2D(size = upsize)(up_conv)
        else: 
            x = Conv2DTranspose(filters = filters,kernel_size=kernel_size,padding=padding)(up_conv)
        x = Concatenate(axis=3)([x,conc_conv])
        x = SpatialDropout2D(0.3)(x)
        x = Conv3x3DB(filters=filters,kernel_size=kernel_size,padding=padding,activation='relu')(x) 
        return x
    return wrapper 

def U_Net(image_size: Tuple[int,int,int] = (128,128,3),
             filters: int = 32,
             kernel_size: Tuple[int,int] = (3,3),
             pool_size: Tuple[int,int] = (2,2),
             padding:str = 'same',
             activation: str = 'relu',
             layers: Union[List[int],int] = 1,
             layers_activation: str = 'sigmoid',
             learning_rate: float = 1e-4,
             loss: str = 'binary_crossentropy',
             metrics: Union[List[str],str] = ['accuracy',],
             upsize: Tuple[int,int] = (2,2),
             upsample: bool = True,
             optimizer = SGD):
    """
        Implementation of U-Net CNN with the help of: 
        https://arxiv.org/abs/1505.04597
        Arguments:
        : image_size - Image size that will be fed to the network, 
        by default (128,128,3)
        : filters - number of convolutional filters in each convolutional block.
        : kernel_size - Tuple reffering to the size of a kernel. 
        : pool_size - Tuple which defines the size of a kernel in pooling layer. 
        : padding - string - whether to use "same" or "valid" padding.
        : activation - activation function, by default "ReLU".
        NOTE: provided string must be an activation function provided by Keras API.
    """
    inputs = Input(image_size)
    conv0 = Conv3x3DB(filters,kernel_size,padding=padding,activation=activation)(inputs)
    pool0 = MaxPoolBlock(pool_size)(conv0)
    conv1 = Conv3x3DB(filters*2,kernel_size,padding=padding,activation=activation)(pool0)
    pool1 = MaxPoolBlock(pool_size)(conv1) 
    conv2 = Conv3x3DB(filters*4,kernel_size,padding=padding,activation=activation)(pool1) 
    pool2 = MaxPoolBlock(pool_size)(conv2) 
    conv3 = Conv3x3DB(filters*8,kernel_size,padding=padding,activation=activation)(pool2)
    pool3 = MaxPoolBlock(pool_size)(conv3) 
    conv4 = Conv3x3DB(filters*16,kernel_size,padding=padding,activation=activation)(pool3) 
    pool4 = MaxPoolBlock(pool_size)(conv4) 
    conv5 = Conv3x3DB(filters*32,kernel_size,padding=padding,activation=activation)(pool4) 
    conv6 = UpSampleBlock(filters*16,kernel_size,padding=padding,upsize=upsize,upsample=True)(conv5,conv4)
    conv7 = UpSampleBlock(filters*8,kernel_size,padding=padding,upsize=upsize,upsample=True)(conv6,conv3) 
    conv8 = UpSampleBlock(filters*4,kernel_size,padding=padding,upsize=upsize,upsample=True)(conv7,conv2)
    conv9 = UpSampleBlock(filters*2,kernel_size,padding=padding,upsize=upsize,upsample=True)(conv8,conv1)
    conv10 = UpSampleBlock(filters,kernel_size,padding=padding,upsize=upsize,upsample=True)(conv9,conv0)
    conv11 = Conv2D(1,1,activation='relu')(conv10)
    out = Flatten(data_format='channels_last')(conv11)
    if isinstance(layers,list):
        for i in range(len(layers)):
            if i == 0:
                output = Dense(layers[i],activation=layers_activation)(out)
            else: 
                output = Dense(layers[i],activation=layers_activation)(output)
    else:
        output = Dense(layers,activation=layers_activation)(out)
    model = Model(inputs,output)
    model.compile(optimizer=optimizer(learning_rate),loss=loss,metrics=metrics)
    model.summary()
    return model