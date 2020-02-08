import keras 
from keras.layers import Dense,Input,Dropout
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.models import Model
from typing import Tuple,Union,List

def custom_ANN(input_shape: Tuple,
               units: Union[List[int],int],
               activation: Union[List[str],str],
               dropout: Union[List[Tuple[bool,int]],int],
               optimizer = SGD,
               learning_rate: float = 1e-4,
               metrics: List = ['accuracy'],
               loss: str = 'binary_crossentropy') -> keras.models.Model:
    """
        This function is responsible for creating a Keras native ANN model in 
        just one line of code with all necessary properties
        Parameters: 
        :input_shape - shape of the data fed into network. 
        :units - List[int] - list of units in each layer of a network.
        :activation - Union[List[str],str] - activation functions in each layer of a network. 
        :dropout - Union[List[Tuple[bool,int]],int] - whether to apply dropout on given layer of network. 
        NOTE: This parameter should be always the same length as units and activation list. Also 
        even if you dont want to apply dropout on some layer, you should still provide a tuple but 
        then you should type ('False',0.0) and then dropout will not be applied.
        :optimizer - type of optimizer that should be used. 
        :learning_rate - learning_rate of an optimizer. 
        :metrics: List - List of accuracy that will be looked during the training phase. 
        :loss - str - defines the loss function.
        Returns: 
        :keras.models.Model - defined keras Model ready to be fit.
    """
    inputs = Input(input_shape)
    if isinstance(units,List): 
        for i in range(0,len(units)):
            if i == 0: 
                out = Dense(units[i],activation=activation[i])(inputs)
                if dropout[i][0] or dropout:
                    out = Dropout(dropout[i][1] if isinstance(dropout,list) else dropout)(out)
            else: 
                out = Dense(units[i],activation=activation[i])(out)
                if dropout[i][0] or dropout:
                    out = Dropout(dropout[i][1] if isinstance(dropout,list) else dropout)(out)
    else: 
        out = Dense(units,activation=activation[0])(inputs)
        if dropout:
            out = Dropout(dropout)(out)
        out = Dense(1,activation=activation[1])(out)
    model = Model(inputs,out)
    model.compile(optimizer=optimizer(learning_rate),loss=loss,metrics=metrics)
    model.summary() 
    return model