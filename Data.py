import librosa as lr
import librosa.display as lrd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import warnings
import wave
import os
from scipy import signal,fftpack
from collections import defaultdict
from typing import List,Union,Tuple,Dict

class DataLoader: 
    """
        This class is responsible for maintaining all processes 
        that are necessary for data to be initially converted 
        into format that is required for models as a input 
        data.
        Parameters: 
        :path - str - path to data csv that should be served in following 
        format: data to audio file and class label for this file
    """
    def __init__(self,path):
        self.path = path
        if not isinstance(self.path,str): 
            raise TypeError("""
                            Path to the file should be an instance of string 
                            and also should directly point to the csv that contains 
                            necessary data to be processed.""")
    def __repr__(self): 
        return f"{self.__class__.__name__}(self.path={self.path})"
    
    def load_data(self,
                sr: int = 44100,
                offset: Union[List,int] = None, 
                duration: Union[List,int] = None, 
                save_path: str = None
                ) -> Tuple[Dict,List]:
        """
          Function is reponsible for loading the data fro given 
          data path and returns a dict containing loaded signal with 
          correspoding data label.
          Parameters: 
          : sr - int - sampling rate of signal, by default is set for 44100 
          which is the most popular sampling rate in acoustic field.
          : offset - Union[List,int] - this parameters defines when the data should
          be loaded from the given signal. It can be set as a list of offsets where each 
          defines when the data should be loaded (in seconds) or as a int.
          : duration - Union[List,int] - parameter defining time (in seconds) at which 
          the signal is being read.
          : save_path - str - file name that will be then saved as a csv with processed data.
          Returns: 
          : Tuple where the first elements represents dict with sampled signal and 
            second elements is a list with corresponding data labels
        """
        self.data = defaultdict(list)
        data_labels = pd.read_csv(self.path,delimiter=';')
        Targets = []
        Path_to_file = []
        i = 0
        if isinstance(offset and duration,list):
            print(f"""Loading data with offsets:{offset}
                      and duration:{duration}""")
            for offsets,durations in zip(offset,duration):
                for audios,target in zip(data_labels['Data'],data_labels['Target']):
                    audio = lr.load(audios,
                                   sr = sr, 
                                   offset = offsets,
                                   duration = durations)
                    self.data[f"{audios}_offset_{offset[i]}_duration_{duration[i]}"].append(audio)
                    Path_to_file.append(f"{audios}_offset_{offset[i]}_duration_{duration[i]}")
                    Targets.append(target)
                i += 1
        else:
            print("Loading data")
            for audios,target in zip(data_labels['Data'],data_labels['Target']):
                    audio = lr.load(audios,
                                   sr = sr, 
                                   offset = offset,
                                   duration = duration)
                    self.data[audios].append(audio)
                    Path_to_file.append(audios)
                    Targets.append(target)
        print("Saving data")
        data_target = self.preparing(self.data)
        processed_data = pd.DataFrame({'Path':Path_to_file,
                                       'Data':data_target,
                                       'Label':Targets})
        processed_data.to_csv(save_path,index=False)
        
        return (self.data,Targets)

    def preparing(self,data: Dict) -> List:
        """ Simple function to converted data from dict returned 
            by load_data function and into list 
            Parameters: 
            : data - Dict - data to be converted
            Returns: 
            : data - List - List of converted data
        """
        data = [item[0][0] for key,item in data.items()]
        return data

    def spectrogram(self,
                   data: Union[Dict,np.ndarray],
                   sr: int,
                   win_length: int = None,
                   hop_length: int = None,
                   save_as_images: bool = None,
                   title: str = None,
                   n_fft: int = 2048,
                   fmin: int = None, 
                   fmax: int = None, 
                   y_axis: str = None, 
                   x_axis: str = None,
                   figsize: Tuple[int,int] = (15,15)
                   ) -> None:
        """
            This function is responsible for applying short time fourier 
            transform on given signal and converts it into spectrogram
            that can be saved as a image (and the used as a input for CNN)
            or just by simply showed just to obtain necessary information 
            about signal. 
            Parameters: 
            : data - Union[Dict,np.ndarray] - data Dict which should contatin 
            sampled signal on which short time fourier transform can be applied
            (so its values should be a type of numpy.ndarray) or just a simple array 
            representing one signal.
            : sr - sampling rate just for spectrogram (the best sampling rate should 
            be the same as sampling rate of given signal)
            : win_length - int - window length of windowing function (by default the windowing function 
            is set to be a hanning window). 
            : hop_length - int - this parameter defines by how much distance the window should move itself after 
            every iteration of short time fourier transform on given signal (if None is passed this parameters 
            is set to value of win_length/4).
            NOTE: this parameter should not be bigger than the window_length.
            : n_fft - int - defining the number of points of short time Fourier transform
            : f_min - int - define the minimum frequency of mel bins (if should be set only if melspectrogram
            is specified)
            : f_max - int - maximum frequency of mel bins
            : y_axis - str - y_axis title of spectrogram. Please note that this parameter should not 
            be a random y_axis. It should be either "log","linear" or "fft"
            : x_axis - str - x_axis title of spectrogram. Should be set to "time"
            : figsize - Tuple[int,int] - defines the width and height of a plot.
            : save_as_images - if specified, the whole dataset of spectrograms will be saved into specified 
            directory in a format of images, so it can be later used a sa input to CNN.
            : title - str - represent the title of spectrogram
            Returns: 
            : None
        """
        plt.figure(figsize=figsize)
        if save_as_images and isinstance(data,Dict):
            for path,value in data.items():
                path = path.split('\\\\')[1]+'.png'
                D = lr.amplitude_to_db(np.abs(lr.stft(value[0][0],n_fft=n_fft,
                                              win_length=win_length,hop_length=hop_length)),ref=np.max)
                lrd.specshow(D,sr=sr,fmin=fmin,fmax=fmax,y_axis=y_axis,x_axis=x_axis,hop_length=hop_length)
                plt.axis('off')
                if os.path.isdir('Images'):
                    pass
                else:
                    print("Making dir: Images")
                    os.makedirs('Images')
                plt.savefig(f'./Images/{path}')
        else:
            warnings.warn("""Please note that passed data into spectrogram should be set 
                             as a numpy.ndarray type otherwise the error will occure.""")
            D = lr.amplitude_to_db(np.abs(lr.stft(data,n_fft=n_fft,
                                                  win_length=win_length,hop_length=hop_length)),ref=np.max)
            lrd.specshow(D,sr=sr,fmin=fmin,fmax=fmax,y_axis=y_axis,x_axis=x_axis,hop_length=hop_length)
            plt.colorbar(format="%+2.0f dB")
            plt.title(title)
    
    def time_course(self,
                    path: str,
                    x_axis: str,
                    y_axis: str,
                    figsize: Tuple[int,int] = (15,15)
                    ) -> None:
        """
            Function reponsible for fast insight of time course of given data 
            before next processing.
            Parameters: 
            : path - str - path to raw data in wav format that should be plotted 
            : x_axis - str - title of x axis
            : y_axis - str - title of y axis
            : figsize - Tuple[int,int] - defining the size of plot 
            Returns: 
            : None
        """
        if not isinstance(path,str): 
            raise Exception("path musi odnosic się do folderu z plikami o rozszerzeniu .wav")
        plt.style.use('ggplot')
        audio = wave.open(path,'r') 
        sygnał = audio.readframes(-1)
        sygnał = np.fromstring(sygnał,"Int16")
        fs = audio.getframerate()
        czas = np.linspace(0,len(sygnał)/fs,len(sygnał))
        plt.figure(figsize=figsize)
        plt.plot(czas,sygnał) 
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(path) 
        plt.show()
    
    def fourier_transform_plot(self,
                             path: str,
                             data: Dict,
                             sr: int,
                             x_axis: str,
                             y_axis: str,
                             title: str,
                             figsize: Tuple[int,int] = (15,15)
                             ) -> None:
        """
           Function responsible for plotting discrete fourier transform 
           on given sampled data, for fast insight. 
           Parameters: 
           : path - str - path of given data to be transformed.
           NOTE: path should be a key on data dict.
           : data - Dict - dictionary with necessary data. 
           : sr - int - sampling rate of the signal. 
           : x_axis - str - title of x plot 
           : y_axis - str - title o y plot 
           : title - str - title of plot 
           : figsize - Tuple[int,int] - Tuple defining size of a plot.
           Returns:
           : None
        """
        n = len(data[path][0][0])
        plt.style.use('ggplot')
        fourier = np.abs(fftpack.fft(data[path][0][0]))/n
        frequency = fftpack.fftfreq(fourier.size)*sr
        plt.figure(figsize=figsize)
        plt.xlim(0,sr/44)
        plt.plot(frequency,fourier)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(title)
        plt.show()