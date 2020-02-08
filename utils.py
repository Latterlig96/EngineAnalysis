import pandas as pd 
import numpy as np 
import librosa as lr
import cv2
from typing import Dict,List,Union,Tuple,Generator
from scipy import fftpack

def clean(path: str
         ) -> Dict:
    """
        Function responsible for converting data
        from csv file into dict where key represents 
        the path to image file and values are corresponding 
        data labels.
        Function is necessary in processing the data in order 
        to provide valid data into CNN.
        Parameters: 
        : path - str - path to csv file that should be converted
        Returns: 
        : data_dict - dict with converted data.
    """
    df = pd.read_csv(path)
    uncleaned_path = df['Path'].tolist()
    targets = df['Target'].tolist()
    data_dict = {}
    for path,target in zip(uncleaned_path,targets):
        path = './Images\\' + path.split('\\\\')[1]
        data_dict[path] = target
    return data_dict

def make_batch(seq,
              size
              ) -> List: 
    return (seq[pos:pos+size] for pos in range(0,len(seq),size))

def generate_data(data: Union[List[str],List[np.ndarray]],
                  labels: Union[Dict,np.ndarray],
                  width: int,
                  height: int,
                  batch_size: int,
                  shuffle: bool
                  ) -> Generator[np.ndarray,np.ndarray,None]:
    """ Keras generator function that generates a bunch of data that 
        is being fed into classifier, by now only designed to generate 
        a bunch of data into CNN model. 
        Parameters: 
        : data - Union[List[str],List[np.ndarray]] - list of strings 
        representing the path to images that should be read.
        : labels - Union[Dict,np.ndarray] - labels for corresponding images. 
        : width - int - width of an image if resize is necessary. 
        : height - int - height of an image. 
        : shuffle - bool - whether to shufle the data before loading.
        Returns: 
        : Generator of both images as a numpy.ndarray and corresponding 
        labels of this images.
    """
    while True: 
        if shuffle: 
            np.random.shuffle(data) 
        for batch in make_batch(data,batch_size):
            X = [cv2.imread(x) for x in batch]
            if width and height:
                X = [cv2.resize(x,(128,128)) for x in X]
                X = [cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in X]
            Y = [labels[x.split('.png')[0]] for x in batch]
            yield (np.array(X),np.array(Y))

def fit_input(data: Union[List[str],List[np.ndarray]],
              labels: Union[Dict,np.ndarray],
              width: int,
              height: int
              ) -> Tuple[np.ndarray,np.ndarray]:
    """ Provides the input data into CNN Keras model from given data.
    Parameters: 
    : data - Union[List[str],List[np.ndarray]] - data of paths to 
    images that should be converted into images. 
    : labels - Union[Dict,np.ndarray] - Dict of data with corresponding 
    data labels where key should be a path to image. 
    : width - int - represents desired width of an image. 
    : height - int - height of an image.
    Returns: 
    : X,y - converted images and their corresponding labels. 
    """
    X = np.array([cv2.imread(x) for x in data])
    if width and height:
        X = np.asarray([cv2.resize(x,(width,height)) for x in X])
        X = np.asarray([cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in X])
    Y = np.array([labels[x.split('.png')[0]] for x in data])
    return (X,Y)

def prepare_for_ANN(data: List,
                    labels: Union[Dict,np.ndarray],
                    width: int,
                    height: int
                    ) -> Tuple[np.ndarray,np.ndarray]:
    """ Easy to use function that converts the data into format 
        which then can be applied as a input data into ANN Keras model. 
        Parameters: 
        : data - List - List of strings that represents path to images.
        : labels - Union[Dict,np.ndarray] - data labels for images. 
        : width - int - if resize is applied, represents the width of an image. 
        : height - int - if resize if applied, represents the height of an image.
        Returns: 
        : Tuple - converted images and corresponding data labels.
    """
    X = np.array([cv2.imread(x) for x in data])
    if width and height: 
        X = np.asarray([cv2.resize(x,(width,height)) for x in X])
        X = np.asarray([cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in X]).flatten().reshape(len(data),-1)
    Y = np.array([labels[x.split('.png')[0]] for x in data])
    return (X,Y)

def prepare_fft_data(data: Dict,
                    labels: List,
                    mode: str = 'stft'
                    ) -> Tuple[np.ndarray,np.ndarray]:
    """ 
        Easy to use function that applies a short time fourier transform 
        or discrete fourier transform on sampled signal, that later can be used 
        as a data into ANN or RNN Keras model.
        Parameters: 
        : data - Dict - Dictionary that holds the data that should be converted. 
        : labels - List - list of corresponding data labels.
        : mode - str - whether to use the short time fourier transform or discrete 
        fourier transform:
        NOTE: if you want to use STFT, then pass "stft" as an argument, otherwise 
        pass 'fft'. 
        Returns: 
        : Tuple - tuple of converted data and corresponding labels.
    """
    ground_truth_labels = []
    fourier_values = []
    if mode == 'fft':
        for label,value in zip(labels,data.values()):
            n = len(value[0][0])
            fourier = np.abs(fftpack.fft(value[0][0]))/n
            ground_truth_labels.append(label)
            fourier_values.append(fourier)
    else:
        for label,value in zip(labels,data.values()): 
            fourier = np.abs(lr.stft(value[0][0],win_length=4096,hop_length=2048,n_fft=4096))
            ground_truth_labels.append(label)
            fourier_values.append(fourier)
            
    return (np.array(fourier_values),np.array(ground_truth_labels))