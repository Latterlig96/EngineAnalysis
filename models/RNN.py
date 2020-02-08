import keras 
from keras.layers import Dense,Input,Dropout,SimpleRNN,Flatten
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.models import Model
from typing import Tuple,Union,List

def custom_RNN(input_shape: Tuple,
               recurrent_units: Union[List[int],int],
               layer_units:Union[List[int],int],
               recurrent_activation: Union[List[str],str],
               layer_activation: Union[List[str],str],
               dropout: Union[List[Tuple[bool,int]],int],
               layer_units_dropout: Union[List[Tuple[bool,int]],int],
               recurrent_dropout: List[float],
               return_sequences: List[bool],
               optimizer = SGD,
               learning_rate: float = 1e-4,
               metrics: List = ['accuracy'],
               loss: str = 'binary_crossentropy') -> keras.models.Model:
    """
        Function provides basic properties that are necessary to create a 
        custom Reccurent neural network in just one line of code. 
        Parameters: 
        :input_shape - Tuple - shape of input data. 
        :reccurent_units - Union[List[int],int] - List of units in reccurent neural network 
        :layer_units - Union[List[int],int] - List of units applied after reccurent phase. 
        :reccurent_activation - Union[List[str],str] - activation functions in reccurent network. 
        :layer_activation - Union[List[str],str] - activation functions for layer units. 
        :dropout - Union[List[Tuple[bool,int]],str] - Tuple of bools and int defines whether 
        to apply dropout in reccurent network or not. 
        :layer_units_dropout - Union[List[Tuple[bool,int]],int] - Tuple of bools and int defines 
        whether to apply dropout in layer units or not. 
        :reccurent_dropout - List[float] - list of floats defining whether to apply reccurent dropout.
        NOTE: Dont use either reccurent dropout and dropout because it will decrease accuracy of training. 
        :return_sequences - List[bool] - List of parameters that defines whether to return the whole sequence 
        of outputs from every each reccurent neuron or just outpout from the last neuron of the current unit.
        :optimizer - type of optimizer used during training phase. 
        :learning_rate - float - defines learning_rate. 
        :metric - kind of metric used during training.
        :loss - str - defines loss function.
        Returns: 
        :keras.models.Model - Defined Keras model ready to be fit.
    """
    inputs = Input(input_shape)
    if isinstance(recurrent_units,list): 
        for idx in range(0,len(recurrent_units)):
            if idx == 0:
                out = SimpleRNN(recurrent_units[idx],
                                activation=recurrent_activation[idx],
                                recurrent_dropout=recurrent_dropout[idx],
                                return_sequences=return_sequences[idx])(inputs) 
                if dropout[idx][0] or dropout: 
                     out = Dropout(dropout[idx][1] if isinstance(dropout,list) else dropout)(out)
            else: 
                out = SimpleRNN(recurrent_units[idx],
                                activation=recurrent_activation[idx],
                                recurrent_dropout=recurrent_dropout[idx],
                                return_sequences=return_sequences[idx])(out)
                if dropout[idx][0] or dropout: 
                    out = Dropout(dropout[idx][1] if isinstance(dropout,list) else dropout)(out)
        out = Flatten()(out)  
        for idx in range(0,len(layer_units)):
            out = Dense(layer_units[idx],activation=layer_activation[idx])(out)
            if layer_units_dropout[idx][0] or layer_units_dropout: 
                 out = Dropout(layer_units_dropout[idx][1] if isinstance(layer_units_dropout,list) else layer_units_dropout)(out)           
    out = Dense(1,activation='sigmoid')(out)
    model = Model(inputs,out) 
    model.compile(optimizer=optimizer(learning_rate),loss=loss,metrics=metrics)
    model.summary() 
    return model