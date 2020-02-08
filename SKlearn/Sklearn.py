import numpy as np 
from typing import Union,List,Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import abc
import warnings

class Skwrapper:
    """ 
        This class provides a simple Sklearn API wrapper
        that is responsible for making a easy one line code predictions 
        of given data and evaluation measured by passed metrics. 
        By now this class provides that kind of functionality for every 
        kind of classifier except of clustering-like algorithms such as 
        KMeans etc.
    """
    def __init__(self,data,labels):
        """
            Parameters: 
            : data - dataset that will be fed into classifiers
            NOTE: this function does not support any kind of test of given 
            format of data yet, so please use right data.
            : labels - data labels.
        """
        self.data = data 
        self.labels = labels
    
    def __repr__(self):
        return f"{self.__class__.__name__}(self.data={self.data)},self.labels={self.labels})"
    
    def split(self,
              state: int = None,
              scale: bool = None,
              test_size: int = None
              ) -> Tuple[np.ndarray,np.ndarray,List,List]: 
        """
            Split the data into training and test set. 
            Parameters: 
            : state - int - defining random_state that will shuffle our data. 
            : scale - bool - whether to use StandardScaler on data or not .
            : test_size - int - defines how big (in percents) our test data should be.
            Returns: 
            : Tuple - tuple representing our train and test data and their corresponding labels.
        """
        X_train,X_test,y_train,y_test = train_test_split(self.data,
                                                         self.labels,
                                                         test_size=test_size,
                                                         random_state=state)
        if scale:
            sc = StandardScaler() 
            X_train_std = sc.fit_transform(X_train) 
            X_test_std = sc.fit_transform(X_test) 
            return X_train_std,X_test_std,y_train,y_test
        else: 
            return (X_train,X_test,y_train,y_test)
    
    def fit(self,
            model,
            metric,
            show_roc: bool = True,
            train: np.ndarray = None,
            test: np.ndarray = None,
            target: np.ndarray = None,
            test_target: np.ndarray = None,
            state: int = None
            ) -> List:
        """
            This function provides one line code easy to use prediction
            of data with given Sklearn model.
            NOTE: this function wraps only models like SVM,MLPClassifier etc.. 
            It will not work on tree-based or ensemble models. 
            Parameters: 
            :model - SKlearn model. It can be sent as a raw callable or 
            as a defined object. 
            :metric - metrics to evaluate score of an model. It can be a one metric
            or a list of metrics.
            :show_roc - bool - whether to plot ROC score or not (by default set as True).
            :train - numpy.ndarray - train set. 
            :test - numpy.ndarray - test set. 
            :target - numpy.ndarray - labels for train set. 
            :test_target - numpy.ndarray - labels for test set. 
            :state - int representing random_state, necessary for shuffling the data. 
            Returns: 
            :List of predictions on test set made by model.
        """
        if type(model) == abc.ABCMeta: 
                    model = model()
        else:
            pass
        clf = model.fit(train,target)
        predictions = clf.predict(test)
        
        if show_roc: 
            self.roc_auc_metric(test_target,predictions)
        if hasattr(metric,'__call__'): 
            print(f"Accuracy {metric(test_target,predictions)}")
        elif isinstance(metric,list): 
            for m in metric: 
                print(f"Metric: {m.__name__} Score: {m(test_target,predictions)}")
        return predictions
    
    def fit_tree_based(self,
                       model,
                       metric,
                       show_roc: bool = True,
                       train: np.ndarray = None,
                       test: np.ndarray = None,
                       target: np.ndarray = None,
                       test_target: np.ndarray = None
                       ) -> List:
        """
            Easy to use function to provide one line code predictions 
            on test set made by any tree-based model from SKlearn API. 
            Parameters: 
            model - Sklearn tree-based model, it can be sent as a raw callable
            or defined object. 
            metric - SKlearn metric that can be sent as a one metric or list of metrics.
            show_roc - bool - whether to plot ROC score or not (by default set as True)
            train - numpy.ndarray - train set. 
            test - numpy.ndarray - test set. 
            target - int - data labels for train set. 
            test_target - int - data labels for test set.
            mission - str
            Returns: 
            :List of predictions made by classifier
        """
        if type(model) == abc.ABCMeta: 
            model = model()
        else:
            pass
        clf = model.fit(train,target) 
        predictions = clf.predict(test) 

        if show_roc: 
            self.roc_auc_metric(test_target,predictions)
        if hasattr(metric,'__call__'):
            print(f"Accuracy {metric(test_target,predictions)}")
        elif isinstance(metric,list): 
            for m in metric: 
                print(f"Metric: {m.__name__} Score: {m(test_target,predictions)}")
        return predictions
    
    def fit_ensemble(self,
                     base_estimator,
                     num_estimators,
                     ensemble_model,
                     metric,
                     final_estimator = None,
                     voting: str = 'hard',
                     show_roc: bool = True,
                     train: np.ndarray = None,
                     test: np.ndarray = None,
                     target: int = None,
                     test_target: int = None,
                     **kwargs
                    ) -> List:
        """
            Function provides one line code predictions of any given 
            SKlearn ensemble model with metrics if provided. 
            Parameters: 
            :base_estimator - SKlearn model that will be used as a base
            estimator in ensemble (it only works if a ensemble model is a BaggingClassifier)
            :num_estimators - number of estimators in ensemble (only if ensemble model is a 
            BaggingClassifier)
            :ensemeble_model - type of ensembling model. 
            :metric - evaluating metric that can be passed as a raw callable or as a list of 
            callables. 
            :final_estimator - if ensemble model is a StackingClassifier, this parameter must be 
            provided so that final predictions will be fed into final_estimators as a input data.
            :voting - str - if VotingClassifier is provided, defines kind of voting(by default is set as hard)
            :show_roc - whether to plot ROC metric 
            :train - numpy.ndarray - defines train set. 
            :test - numpy.ndarray - defines test set. 
            :target - numpy.ndarray - data labels for train set.
            :test_target - numpy.ndarray - data labels for test set.
            Returns:
            :List of predictions made by classifier
        """
        if isinstance(base_estimator,list): 
            models = [] 
            for model in base_estimator:
                if type(model) == abc.ABCMeta: 
                    model = model()
                else:
                    pass
                models.append((str(model.__class__.__name__),model))
            if str(ensemble_model.__name__) == 'VotingClassifier':
                clf = ensemble_model(estimators=models,voting=voting).fit(train,target) 
                predictions = clf.predict(test)
            else: 
                warnings.warn("Unfortunately it wont work, the problem lays in Sklearn API")
                clf = ensemble_model(estimators=models,final_estimator=final_estimator)
                predictions = clf.predict(test)
        elif ensemble_model.__name__ == 'BaggingClassifier':
            clf = ensemble_model(base_estimator=base_estimator,
                                n_estimators=num_estimators,**kwargs).fit(train,target)
            predictions = clf.predict(test) 
        else: 
            clf = ensemble_model(n_estimators=num_estimator,**kwargs).fit(train,target) 
            predictions = clf.predict(test)
            
        # This section covers only metric processing
        if show_roc: 
            self.roc_auc_metric(test_target,predictions)    
        if hasattr(metric,'__call__'): 
            print(f"Model accuracy of metric {metric(test_target,predictions)}")
        elif isinstance(metric,list): 
            for m in metric: 
                print(f"Metric: {m.__name__} Score: {m(test_target,predictions)}")
        return predictions
    
    def roc_auc_metric(self,
                       y_true: Union[np.ndarray,List] = None,
                       predictions: Union[np.ndarray,List] = None,
                       figsize: Tuple[int,int] = (10,10)
                       ) -> None:
        """
            Function responsible for plotting ROC metric.
            Parameters: 
            :y_true - Union[numpy.ndarray,List] - list of ground_truth labels. 
            :predictions - Union[numpy.ndarray,List] - predictions made by model. 
            :figsize - Tuple[int,int] - defines a size of a plot.
        """
        print(f"ROC metric: {roc_auc_score(y_true,predictions)}")
        fpr,tpr,thresholds = roc_curve(y_true,predictions)
        plt.figure(figsize=figsize)
        plt.style.use('ggplot')
        plt.plot([0,1],[0,1])
        plt.plot(fpr,tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Metric')
        plt.legend(loc="lower right")
        plt.show()