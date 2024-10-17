import numpy as np

class BaseEstimator:
    """
        Base class for all estimators
        
        Parameters
        ----------
        y_required : bool, default = True
            Determines whether the data model requires label focus (y)
        fit_required : bool, default = True
            Determines whether the fit function must be called before calling the predict function
    """
    y_required = True
    fit_required = True
    
    def __init__(self):
        pass

    def _setup_input(self, X, y = None):
        
        
        # make sure X is a numpy array
        if not isinstance(X, np.ndarray):
           X = np.array(X)
        
        # check if X is empty
        if X.size == 0:
            raise ValueError("X is empty matrix")
        
        # check if X is 1D or N-D array
        if X.ndim == 1:
            # determine that there is 1 unique sample 
            # get the feature count from the size of the array X 
            self.n_samples, self.n_features = 1, X.shape[0]
        else :
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
        
        # store the input data
        self.X = X
        
        # check if y is required
        if self.y_required:
            # check if y is None     
            if y is None:
                raise ValueError("Missed required argument y")
            
            # make sure y is a numpy array
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            
            # check if y is empty
            if y.size == 0:
                raise ValueError("y is empty matrix")
        
        # store the label data
        self.y = y
    
    def fit(self, X, y = None):
        # setup input data
        self.setup_input(X, y)
        return self
    
    def predict(self, X = None):
        # make sure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # check if training has been done or not required calling fit be
        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else :
            raise ValueError("You must call 'fit' function before 'predict'")
        
    def _predict(self, X= None):
        raise NotImplementedError()