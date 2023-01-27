#from keras.optimizers import Optimizer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import t
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import numpy as np
import pandas as pd
import copy

#from keras.models import Model, clone_model

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm


def ci_percentile(errors, alpha, method="linear"):
    """Compute confidence interval based on the Numpy percentile method.

    Args:
        errors (Numpy Array): Array of errors
        alpha (Numeric): number from 0 to 1 indicating the confidence interval spread, e.g., 0.05 (95%)
        method (str, optional): Defaults to "linear", method = one of "inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased"
    Returns: 
        error_lower, error_upper
    """

    percentile_upper = 100 * (alpha / 2 )
    percentile_lower = 100 - 100 * (alpha / 2 )
    
    error_lower = np.percentile(errors,percentile_upper,method=method)
    error_upper = np.percentile(errors,percentile_lower,method=method)
    
    return error_lower, error_upper

def ci_tdistribution(errors, alpha):
    """Compute t-distribution confidence intervals from the provided errors and alpha

    Args:
        errors (Numpy Array): Array of errors
        alpha (Numberic): number from 0 to 1 indicating the confidence interval spread, e.g., 0.05 (95%)

    Returns: 
        error_lower, error_upper
    """
    
    n = len(errors)
    s = np.std(errors, ddof = 1)  # sample std dev, divisor= N-ddof, delta degrees of freedom ... = N-1 for sample std dev
    t_critical = t.ppf(1-alpha/2, df = n-1) # account for both tails, prob of each tail is alpha/2
    e_mean = errors.mean()
    SE = t_critical * s / np.sqrt(n)
    
    error_lower = e_mean - SE
    error_upper = e_mean + SE

    return error_lower, error_upper

def forecast_confidence(df,alpha, Nhorizon = 1, error="error", method="linear", verbose=False, debug=False):
    """Manage the computation of the confidence interval based on the selected method, from the folowwong choices,
    * numpy percentile 
    * t-statistics
    * minmax observed error

    Args:
        df (DataFrame): DataFrame containing error column from which the confidence interval is computed.
        alpha (Numeric): Number from 0 to 1 defining the confidence error spread, for ecample 0.95 (95%).
        Nhorizon (int, optional): _description_. Defaults to 1.
        error (str): Column which contains the error values from which the confidence interval is computed.
        method (str): Method to cacluate the confidence interval. Defaults to "linear". Defaults to "linear" from the numpy percentile function. Choices are as follows. 
            * mumpy.percentil() function, method = one of -  "inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased"
            * "minmax" - the min and max values of observed errors
            * "tdistribution" - compute the t-distribution confidence interval
        verbose (bool, optional): _description_. Defaults to False.

    Returns: 
        Dataframe with error_lower and error_upper
    """
    
    df_error = pd.DataFrame(df[error])
    df_error = df_error.dropna() # only keep the horizon forecast periods, all others will be NA
    df_error["horizon_id"] = 1
    df_error["horizon_id"] = df_error["horizon_id"].cumsum() - 1 # 0, 1, 2, 3 ...
    df_error["horizon_id"] = df_error["horizon_id"]//Nhorizon  # horizon_id = 0 0 0 , 1 1 1,   ... 
    df_error["horizon"] = 1
    df_error["horizon"]=df_error.groupby("horizon_id")["horizon"].cumsum() # horizon = 1 , 2, 3 , 1, 2, 3, 1 , 2, 3, ...
    
    if verbose or debug == True:
        print(f'forecast_confidence: Nhorizon = {Nhorizon}')
        print(f'forecast_confidence: method = {method}')
        
    if debug == True:
        print(f'forecast_confidence: df (prior to joining error_lower and error_upper')
        display(df)
    
    nh_errors_lowerupper =[]
    for nh in np.arange(1,Nhorizon + 1):
 
        errors_nh = df_error[df_error["horizon"]==nh][error].values
        nh_idx = df_error[df_error["horizon"]==nh][error].index
        
        percentile_methods = (  "inverted_cdf", "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf",
                                "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear",
                                "median_unbiased ", "normal_unbiased")
 
  
        error_lower, error_upper = 0,0
        if method == "tdistribution":
            error_lower, error_upper = ci_tdistribution(errors_nh, alpha)
        elif method == "minmax":
            error_lower,error_upper = errors_nh.min(), errors_nh.max()
        elif method in percentile_methods:
             error_lower,error_upper =  ci_percentile(errors_nh, alpha, method=method)
        else:
            pass
                    
        nh_errors_lowerupper = nh_errors_lowerupper + [(error_lower,error_upper)]
        df_error.loc[nh_idx,"error_lower"] = error_lower
        df_error.loc[nh_idx,"error_upper"] = error_upper

        if debug == True: 
            print(f'forecast_confidence: horizon nh = {nh}')  
            print(f'forecast_confidence: nh_idx = {nh_idx}, errors = {errors_nh}')
            print(f'forecast_confidence: interval, error_lower ={error_lower}')     
            print(f'forecast_confidence: error_upper ={error_upper}')    
            
    df = df.join(df_error[["error_lower","error_upper"]])
    
    if debug == True:
        print(f'forecast_confidence: df (after joining error_lower and error_upper')
        display(df)
        print(f'forecast_confidence: nh_errors_lowerupper = {nh_errors_lowerupper}')
    
    return df, nh_errors_lowerupper

def min_func(x,minvalue):
    """return x if greater than minvalue, otherwise return minvalue. Vectorized into min_vfunc for use by numpy.

    Args:
        x (numeric): number to compare to minvalue
        minvalue (numeric): minimum allowed value

    Returns:
        returns x if greater than minvalue, otherwise returns minvalue
    """
    if  ~np.isnan(x):
        x = x if x > minvalue else minvalue
    return x

def max_func(x,maxvalue):
    """return x if less than maxvalue, otherwise return max value. Vectorized into max_vfunc for use by numpy array.

    Args:
        x (numeric): number to compare to max value 
        maxvalue (numeric): maximum allowed value

    Returns:
        returns x if greater than minvalue, otherwise returns minvalue
    """
    if ~np.isnan(x):
        x = x if x < maxvalue else maxvalue
    return x

# Numpy vectorized functions
min_vfunc = np.vectorize(min_func)
max_vfunc = np.vectorize(max_func)

# Transformers
#    BaseEstimater ... get_params(), set_params() methods
#    TransformerMixin ... fit_transform() method

def nlag_covars(df:object,covars:list, N_lags:int) -> object:
    """ add covariat lags (nlag columns) to DataFrame

    Args:
        df (object): DataFrame containing covariates
        covars (list): list of columns (covariates) that will be added to the input dataframe
        N_lags (int): add df([col]).shift(i) from i to Nlags for each col in co_vars

    Returns:
        DataFrame of XY variables with the addition of lagged variables.
    """
    
    # co_vars := auto regressive varibles
    
    dfXY = df.copy()
    
    # ensure that co_vars is iterable
    covars = [covars] if isinstance(covars,str) else covars

    for n in np.arange(1,N_lags+1):
        cv_shift_vars = []
        for cv in covars:
            cv_shift_var = cv+"_m"+str(n)
            cv_shift_vars.append(cv_shift_var)
        dfXY[cv_shift_vars] = dfXY[covars].shift(n)
        
    return dfXY

# covarlags tranformer
class covarlags(BaseEstimator,TransformerMixin):
    """ Transfomer for creating lagged variables. The transformer
    includes fit() and transform() , and fit_transform() operations (by extension of TrasnformerMixin).
    The transofrm retains a memory of Nlags+1 columns.
    
    **__init__**
    
    Args:
        covars (list): list of column names representing covariates.
        Nlags (int): number of lags to implement for each covariate.
    
    """
    def __init__(self, covars=None, Nlags=1):
        self.covars=covars
        self.Nlags = Nlags
        self.Nmemory = Nlags+1
        self.dfmemory = None
        assert self.Nlags >=0 ,f'Nlags = {self.Nlags} not allowed. Nlags must be >= 0'
        
    def fit(self,df):
        
        self.dfmemory = df.tail(self.Nmemory) if df.index.size > self.Nmemory else df.index.size
        return self
    
    def transform(self, df=None, Nout=None, dfnewrows=None, debug=False):
        # if df not spefified then transform dfmemory
        # add new row(s) and update dfmemory
        """Create lagged variables from covariates.

        Args:
            df (DataFrame, optional): _description_. Defaults to None.
            Nout (int, optional): Number of output rows. Defaults to None.
            dfnewrows (DataFrame, optional): new row to append to the DataFrame. The DataFrame memory will be updated with the last Nlags+1 rows. Defaults to None.
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            DataFrame with the addition of lagged variables
        """
        
        if not isinstance(df,pd.DataFrame):
            df = self.dfmemory
            #print("covarlags.transform: dfmemory before new rows =")
            #display(df.tail())
            if isinstance(dfnewrows,pd.DataFrame):
                df = pd.concat([df,dfnewrows])
                self.dfmemory = df.tail(self.Nmemory)    
            
            if debug == True:    
                print("  covarlags.transform: new row dfmemory.tail()")
                display(df.tail(3))
        
        dfout = df.copy()
        dfout = nlag_covars(dfout,self.covars, self.Nlags)
        #  do not delete first Nlag rows it will be done above to avoid cumulative  deleting due to derived variables ... dfout = dfout.iloc[self.Nlags:] 
        
        return dfout if Nout==None else dfout.tail(Nout)
    
    def get_last_dfrow(self): # last row from dfmemory ... corresponds to the most recent past
        """
        Returns:
            the last row of the saved DataFrame
        """
        return self.dfmemory.tail(1)
    
    def set_last_y(self,y,yvalue, debug=False):
        """Update the value of the column "y" corresponding to the last row of the saved DataFrame.

        Args:
            y (string or list): column name(s) of the last row to update with the values yvalue
            yvalue (Numeric or list of Numerics): value(s) corresponding to "y"

        """
        #last_idx = self.dfmemory.index[ self.dfmemory.index.size - 1 ] # this method can cause subtle mis matches in index types 
        last_i =  self.dfmemory.index.size - 1 
        self.dfmemory[y].iloc[last_i] = yvalue
        if debug == True:
            print("  covarlags.set_last_y, dfmemory.tail()")
            display(self.dfmemory.tail())
        return None
    
# minmaxstd scaler        
class minmaxstd_scaler(BaseEstimator,TransformerMixin):
    """The minmaxstd scaler is a transformer that scales ML features according to the SKLEARN minmax scaler (default) and/or the SKlearn StandardScaler.
    The class implements fit() and Transform(). The class extends BaseEstimator and TransformerMixin which extends the fit_transorm() operation.
    
    **__init()__**
    
    Args:
        * mms_cols(List): list of columns that will be transformed by the MinMaxScaler(). If mms_cols = "all" then all columns are transformed by the MinMaxScaler().
        * ss_cols(List): list of columns that will be transformed by the StandardScaler(). ss_cols takes prededance over ss_cols.
        * donotscale_cols (List): list of columns to ignore (not scale).
        
    """
    def __init__(self, mms_cols=None, ss_cols=None, donotscale_cols=None):
        self.mms_cols = mms_cols
        self.ss_cols = ss_cols
        self.donotscale_cols = donotscale_cols
        self.mmscaler = None
        self.sscaler = None

    def fit(self, df):
        """Fit the scaler to the input DataFrame.

        Args:
            df (DataFrame): input DataFrame with columns to be fitted.

        """
        # mms scaler fit
        if self.mms_cols != None:
            scaler = MinMaxScaler()         
            if (self.mms_cols == "all") & (self.ss_cols != "all"): 
                self.mmscaler = scaler.fit(df) 
            elif self.mms_cols != "all":
                self.mmscaler = scaler.fit(df[self.mms_cols])
        # standard scaler fit        
        if self.ss_cols != None:
            scaler = StandardScaler()
            if self.ss_cols == "all":
                self.sscaler = scaler.fit(df)  # used pre-fitted scaler when predicting
            else:
                self.sscaler = scaler.fit(df[self.ss_cols])   
                   
        return self 
    
    def transform(self, df):
        """Transform the input DataFrame (df)

        Args:
            df (DataFrame): input DataFrame with columns to be scaled.

        Returns:
            DataFrame with scaled columns.
        """
        df_scaled = df
        if self.mms_cols != None:
            if (self.mms_cols == "all") & (self.ss_cols != "all"): 
                df_scaled = self.mmscaler.transform(df)
                df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)
            elif self.mms_cols != "all":
                df_scaled[self.mms_cols] = self.mmscaler.transform(df_scaled[self.mms_cols])

        if self.ss_cols != None:
            if self.ss_cols == "all":
                df_scaled = self.sscaler.transform(df) # returns Numpy array
                df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)
            else:
                df_scaled[self.ss_cols] = self.sscaler.transform(df_scaled[self.ss_cols]) # Numpy nd-array into cols           
        
        if self.donotscale_cols != None:
            df_scaled[self.donotscale_cols] = df[self.donotscale_cols]
        
        return df_scaled
    

def train_test_predict(dfXY:DataFrame, y:list,  model:object, model_type="sk", cm_params=None,  scale_params = None,
                    swin_params=None,  tf_params=None, fit = True, predict=False, pred_params = None, verbose=False , debug=False) -> list:
    """The train_test_predict() function is the work horse producing the fit (train and test) and predictions. 
    During the fit operation test_train_predict() receives the input observations (dfXY) and fits the make lags transform, scaler transform, and derived
    attriutes transform as required by the input paramters. Train_test_predict manages the transformation of the input data for compatiability
    with the specified model (Classical, TensorFlow, or SKLearn), fits the the model to the training data, and predicts on the test dataset. 
    
    When fit = True, test_train_pridict() fits the model and transforms including make_lags, derived_attributes transform, and scaler transform. 
    The model is fit to the training set and predicts over Nhorizon according to 
    recursive (Nhorizon x 1-step recursive predictions) or direct forecasting (Nhorizon models) over Nhorizon time steps. 
    
    The out-of-sample train-test methodology works as follows. The first training and test window (before sliding) has a training set comprised of total observations - Ntest and test set size
    Ntest. The model is retrained every Nhorizon steps, where Nhorizon defaults to 1. 
    After the first train/test operation, the training set window slides forward by Nhorizon observations. The training set increases by Nhorizon observations, and the 
    test set decreases by Nhorizon observations. Predictions (y_pred) and actual (y_test) are returned based on each train-test window. 
    
    The fitted transformers, make_lags, scaler, and make_derived_attributes, transform the data during test or prediction operations. 
    The transformers generate lagged variables, derived variables, and, together with exogenous variables, are scaled. These are subsequently 
    input to the fitted model to make a prediction.
     
    The last fit corresponds to all observation in the dfXY input DataFrame. 
    The fitted model, make_lags transform make_derived_attributes, and scaler transform are returned in addition to test predictions, and prediction index (y_pred_idx) relating the prediciton to the input DataFrame.

    When predict = True, test_train_predict() recieves the pred_params dictionary with the fitted make_lags, make_derived_attributes, and scaler transform. Nperiod predictions
    are made according to the fitted model, recursive or direct forcast. For direct forecast Nperiods must be <= Nhorizon, since there are Nhorizon trained models.
    
    Args:
        dfXY(DataFrame): DataFrame with dependent and independennt variables.
        y (string, list): target variable string, or list of variables.
        model (object): reference/pointer to the forecast model
        model_type (str, optional): type of forecast model "cm" (classical), "sk" (SK Learn), or "tf" (TensorFlow). Defaults to "sk".
        swin_params (Dictionary): sliding window parameters. See sforecast documentation.
        cm_params (Dictionary, optional): Classical model parameters. See sforecast documentation.
        scaler_params (Dictionaryj): scaler teransform including MinMax and StandardScaler. See sforecast documentation.
        tf_params (Dictionary, optional): TensorFlow paramters. See sforecast documentation
        fit (Booelan): Default = True. Fit a model with sliding forecast (past training and future test) observations. When fit = False, then predict using the previously trained model. Variables will be scaled with the fitted scalers (MinMax or Standard).
        predict (Boolean): Default = False. Predict based on the the fitted model and transforms (make_lags, scaler, derived_attributes).
        pred_params (Dictionary, optional): This input is required when predict = True, otherwise it is ignored. The pred_params dictionary contains the fitted transorms (make_lags, scaler, make_attributes_transform), which are applied during the predict operation. See sforecast for further docuentation.
        verbose (bool, optional): Defaults to False.
        
    Returns: variables
        * y_pred_nda (Numpy n-dimensional arrray): predicted values  
        * y_test_nda (Numpy n-dimensional arrray): test_values (i.e., truth, actual observations). y_test_nda = None when predict = True.
        * y_pred_idx (List): list of integers or timestamps corresponding to the location of the prediciton relative to the input DataFrame.
        * dfXY_train (DataFrame): fitted dfXY DataFrame including scaled variables exogvars, covariates, and derived attributes.
        * m (model): ML forecast model. When model_type = "sk" or "tf" and "cm" AUTO_ARIMA , m corresponds to the fitted model, after the last traiing.
        * model_fit: Fitted model for the case of model_type == "cm" ARIMA and ARIMAX , otherwise = None.
        * i_initial_pred: i-th (iloc) index corresponding to the first prediction
        * make_lags (transformer): fitted transformer for creating lagged variables.
        * scaler (transformer): fitted scaler for MinMaxScaler and or StandardScaler transformations.
        * make_derived_attributes: fitted make derived attributes transformer.
        * history_i: TensorFlow training history (initial training)
        * history_t: TensorFlow training (tunning) history corresponding to the final (last) training.
        
        return:  
            * if model_type == "cm":  
                return = y_pred_nda, y_test_nda, y_pred_idx, dfXY_train, m, model_fit, i_initial_pred, make_derived_attributes, make_lags, scaler  
            * if model_type == "sk":  
                return = y_pred_nda, y_test_nda, y_pred_idx, dfXY_train, m, i_initial_pred, make_derived_attributes, make_lags, scaler  
            * elif model_type =="tf":   
                 return = y_pred_nda, y_test_nda, y_pred_idx, dfXY_train, m, history_i, history_t,i_initial_pred, make_derived_attributes, make_lags, scaler  
        
    """
    
    if fit ==True:
        assert predict == False, f'predict must be False if fit == True'
        
    assert (fit == True or predict == True), f'Either fit or predict must be True'
    
    # ensure y is a list
    y = [y] if isinstance(y,str) else y
    
    # unpack swin_params
    catvars=swin_params["catvars"]
    exenvars=swin_params["exenvars"]
    exogvars=swin_params["exogvars"]
    covars = swin_params["covars"]
    Ncovars = len(covars)
    mms_cols = scale_params["mms_cols"]
    ss_cols = scale_params["ss_cols"]
    Nhorizon = swin_params["Nhorizon"]
    Nlags=swin_params["Nlags"]
    Ntest = swin_params["Ntest"]
    Ntest = swin_params["Ntest"]
    idx_start = swin_params["idx_start"]
    horizon_predict_method = swin_params["horizon_predict_method"]
    derived_attributes_transform = swin_params["derived_attributes_transform"]
    derived_attributes = swin_params["derived_attributes"]
    Ndclip = derived_attributes_transform().get_Nclip() if derived_attributes_transform != None else 0
    
    # flat exenvars
    if  exenvars!=None:
        exenvars_flat = [ex for exen in exenvars for ex in exen] if not isinstance(exenvars[0],str)  else exenvars
                     
    # unpack predict variables
    if predict == True:
        index_pred = pred_params["index"]
        dfXexogs = pred_params["dfexogs"]
        Nperiods = pred_params["Nperiods"]
        make_lags = pred_params["make_lags"]
        make_derived_attributes = pred_params["make_derived_attributes"]
        scaler = pred_params["scaler"]
        
        #print("test_train_predict: predict, index_pred =", index_pred)
        #print("test_train_predict: predict, make_lags =", make_lags)
    
    if verbose == True or debug == True:
        print("test_train_predict: fit, Nhorizon =",Nhorizon)
        print("test_train_predict: covars =", covars)
        print("test_train_predict: exenvars =", exenvars)
        print("test_train_predict: catvars =", catvars)

    # initialize Tensorflow variables ... needed for return
    history_i = None # tensorflow initial training history, first fit
    history_t = None # tensorflow tuning/training history second, third, ... fit
    model_fit = None # for arima and sarima models (not autoarima)
    
    #### Loop Variables ... e.g., i_first, i_ilast ...
    
    if fit == True: # loop variables for fit == True
    
        ### dfXY  only keep up to the last fit ... toss the rest
        i_lastrow =  (dfXY.index.size - 1)  # default lastrow  of dataframe
        i_lastrow =  (dfXY.loc[:idx_start].index.size - 1) if idx_start != None else i_lastrow #      
        dfXY = dfXY[:i_lastrow + 1] # only keep up to last pred row, +1 to include i_last otherwise it will be left out since numbering starts at 0 ... this is necessary if start wiith idx_start, or i_sstart

        ### ith loop beginning and end 
        
        cm_model = cm_params["model"] if model_type == "cm" else None # unpack cm_model for convenience

        if model_type == "cm" and (cm_model == "arima" or (cm_model=="sarimax" and derived_attributes_transform == None) or (cm_model=="auto_arima" and derived_attributes_transform == None)) :
            lags = False
            derived = False
            Nclip = 0
        else:
            lags = True
            derived = True if derived_attributes_transform != None else False
            Nclip = Nlags if Nlags > Ndclip else Ndclip  # discard NaN initial rows due to Nlag shifts and derived_attributes 
             
        N_loop_step = Nhorizon  
        autoarima_nperiods = Nhorizon
        i_initial_pred = i_lastrow - Ntest + 1 - Nclip if Ntest > 0 else i_lastrow - Nclip + 1
        i_loop_first = i_initial_pred  #  
        i_lastrow = dfXY.index.size - 1 - Nclip #  new i_lastrow account for NaN clip (Nclip)
        i_loop_last = i_lastrow + 1 # # ... allows training on the last row  
                                    # example: i_lastrow = 30 , i_loop_last = 31, last train on i = 30 

        #i_last_predict ... ith row of last prediction. Only autoarima without derived does multi-step prediction
        if derived == False and  model_type == "cm" and cm_params["model"] == "auto_arima" : # all other cases besides arima non-derived manage need covarlags and cannot do N-step prediction
            i_last_predict = i_lastrow  - ( Nhorizon - 1 )
        else:
            i_last_predict = i_lastrow # last prediction of the fit process (i.e., training and test)

        i_last_fit = i_loop_last   # train on the last observation row ... prediction method will be based on last row fit ... afer fit phase, then predict forward
        
        if debug == True:

            print("test_train_predict: lags =",lags)
            print("test_train_predict: derived =",derived)
            print("test_train_predict: Nclip =", Nclip)
            print("test_train_predict: dfXY size, pre clip = ",dfXY.index.size)
            print("test_train_predict: dfXY size, post clip = ",dfXY.index.size - Nclip)
            print("test_train_predict: i_lastrow ", i_lastrow)  # Nclip already subtracted above
            print("test_train_predict: i_last_predict ", i_last_predict)  # Nclip already subtracted above
            print("test_train_predict: i_initial_predt =", i_initial_pred)
            print("test_train_predict: i_last_fit =", i_last_fit)
            print("test_train_predict: i_loop_first =", i_loop_first)
            print("test_train_predict: i_loop_last =", i_loop_last)
            print("test_train_predict: N_loop_step = ",N_loop_step)
    
        # initialize additional varialbels 
        make_lags = None  # default ot none ... None for cm models, not none for sk and tf models
        make_derived_attributes = None # default to noe ...None for cm models, not none for sk and tf models
        scaler = None # # default to noe ...None for cm models, not none for sk and tf models
        
        y_pred_nda = None
        y_test_nda = None
        y_pred_idx = None
        

        # predict_cnt 
        predict_cnt = 0

    if predict == True: # loop parameter for predict == True
        i_initial_pred = 0
        Nclip = 0
        i_loop_first = i_initial_pred # 
        i_last_predict = i_loop_first + Nperiods - 1 # Nperiods, starting at i_loop_first
        i_loop_last = i_last_predict                 # Nhorizon predictions/forecasts starting at i_loop_first
        lags = True if make_lags != None else False
        derived = True if make_derived_attributes != None else False
        m = model
        model_fit = m 
        N_loop_step = Nhorizon if horizon_predict_method == "multi_step" else 1
        autoarima_nperiods = Nperiods

        if debug == True:
            print("test_train_predict: predict, index_pred =",index_pred)
            print("test_train_predice: predict, Nperiods =", Nperiods)
            print("test_train_predict: i_loop_first =",i_loop_first)
            print("test_train_predict: i_last_predict =",i_last_predict)
            print("test_train_predictl: i_loop_last =",i_loop_last)
            print("test_train_predict: N_loop_step = ",N_loop_step)
            
        # predict_cnt 
        predict_cnt = 0
    
    ####### LOOP ... train/test/predict  #########
    dfXY_train = None
    outer_loop_break = False
    for i in np.arange(i_loop_first, i_loop_last + 1, N_loop_step):  # +1 so last i will be i_loop_last
        
        if outer_loop_break == True:
            break
        outer_loop_break = False
        
        if debug == True: 
            print("\ntrain_test_predict: outer-loop, i_loop_last =", i_loop_last,",i =", i)
                    
        if fit == True:  
            # dfXY observations ... prepare dfX, dfY, dfXtrain, dfXexogs, dfXlags ...
            # dfXY_train before covarlags ... before clip
            dfxy = dfXY
            dfy = dfXY[y]
            dfXY_train = dfXY.iloc[:i + Nclip] # add Nclip since adjusted for Nclip above
            dfY_train = dfy.iloc[:i + Nclip] # add Nclip since adjusted for Nclip above

            
            #### Covarlags
            if lags: #  need lags for cm if have derived_attributes
                make_lags = covarlags(covars=covars,Nlags=Nlags)
                if model_type == "tf":
                    if catvars != None:
                        dfXcats_train = dfXY_train[catvars] 
                        _dfXY_train = dfXY_train.drop(catvars, axis =1 )
                        dfXY_train=make_lags.fit_transform(_dfXY_train) #
                    else:
                        dfXY_train=make_lags.fit_transform(dfXY_train)
                        
                else:
                    dfXY_train=make_lags.fit_transform(dfXY_train)

            # dfX_train ... after make lags 
            dfX_train = dfXY_train.drop(covars,axis=1)
            
            #### Derived Variables
            # dfXexen
            #   ouput of derived variables contains 
            #     endogenous derived variables 
            #     exogenous variables
            #     covarlags
            #     y variables needed for derived ... later drop the y variables ... will only keep past derived variables to prevent leakage
            if derived:
                make_derived_attributes = derived_attributes_transform()
                dfX_train = make_derived_attributes.fit_transform(dfX_train) # dfXexen contains exogenous and endogenous (derived) variables
                dfXY_train[derived_attributes]=dfX_train[derived_attributes].values # put derived attributes id dfXY for consistency
                
                
            if lags or derived:
                #### delete NaN rows
                dfy= dfy.iloc[Nclip:] # 
                dfxy = dfxy.iloc[Nclip:]
                dfXcats_train = dfXcats_train.iloc[Nclip:] if model_type == "tf" and catvars != None else None
                dfXY_train = dfXY_train.iloc[Nclip:]
                dfY_train = dfY_train.iloc[Nclip:]
                dfX_train = dfX_train.iloc[Nclip:]
                
            else:
                #### delete NaN rows
                dfy= dfy # 
                dfxy = dfxy
                dfXcats_train = dfXcats_train if model_type == "tf" and catvars != None else None
                dfXY_train = dfXY_train
                dfY_train = dfY_train
                dfX_train = dfX_train
                      
            # Scale
            # do not scele cm models
            # tf scaled below
            # sk models scale of dfX_train
            #  covar lags (not unlagged) and exogenous variables are scaled ..  
            #  do not scale catvars
            #  y (target variables) not scaled
            if  model_type == "sk" or model_type == "tf":
                scaler=minmaxstd_scaler(mms_cols=mms_cols,ss_cols=ss_cols)
                dfX_train = scaler.fit_transform(dfX_train)
                dfXY_train = dfXY_train[covars].join(dfX_train) # add covars back to dfX_train to keep dfXY_train consistent
            
            if debug == True:
                print("test_train_predict: dfX_train")
                display(dfX_train.tail(3))
                print(f'test_train_predict: dfX_train.shape = {dfX_train.shape}')
            
            # this Y_train only used by cm ... sk and tf create their own Y_tran based on covars
            # Y_train values  ... single_step and multi_step
            # if multi_step ...  Y_train list for each time-step within Nhorizon ...
            Y_train = dfY_train[y[0]].values           # sinlge_step prediction simply needs Y_train ...  y is a list so take the first element 
            if horizon_predict_method == "multi_step":  # multi_step has family of Y_train values for fit below
                Y_train = [Y_train]
                for n in np.arange(1,Nhorizon):
                    Y_train = Y_train + [dfY_train[y[0]].shift(-n).values] # shift -n ... will contain n NaN s at the end ... clip off below 
            if model_type == "cm" and  debug == True:
                print("test_train_predict: dfY_train")
                display(dfY_train.tail())
                print('test_train_predict: dfY_train.shape = {dfY_train.shape} ')
                
            # dfXexen_train    
            dfXexen_train = dfX_train[exenvars_flat] if exenvars != None else None

            #######################
            ###### fit models #####
            if model_type == "cm" and cm_params["model"] == "arima": # fit arima

                if debug == True: 
                    print("train_test_predict: arima fit i =", i)
                
                m = ARIMA(Y_train, order=cm_params["order"])
                model_fit = m.fit()
                
            if model_type == "cm"  and cm_params["model"] == "sarimax": # fit sarimax
                if exenvars == None:
                    if debug == True: 
                        print("train_test_predict: sarimax fit i =", i)
                     
                    m = SARIMAX(Y_train, order= cm_params["order"] , seasonal_order = cm_params["seasonal_order"],
                                enforce_stationarity = cm_params["enforce_stationarity"],
                                enforce_invertibility = cm_params["enforce_invertibility"] )
                else:

                    if debug == True: 
                        print("train_test_predict: sarimax fit, with exenvars, i =", i)
                    
                    Xexen_train = dfXexen_train.values if exenvars != None else None
                    
                    m = SARIMAX(Y_train, exog = Xexen_train, order= cm_params["order"] , seasonal_order = cm_params["seasonal_order"],
                                enforce_stationarity = cm_params["enforce_stationarity"],
                                enforce_invertibility = cm_params["enforce_invertibility"] )
                
                model_fit = m.fit(disp=False)
            
            if model_type == "cm" and cm_params["model"]== "auto_arima": # fit pdarima model
                
                if debug == True: 
                    print("train_test_predict: autoarima fit, i =", i)
                
                # auto_arima initialize and fit model in one step
                if exenvars == None: # pdarima w/o exenvars
                    
                    if debug == True:
                        print("test_train_predict: auto_arima fit w/o exenvars")
                    m = pm.auto_arima(Y_train, 
                        start_q = cm_params["start_q"],
                        start_p = cm_params["start_p"],
                        d = cm_params["d"],
                        test = cm_params["test"], # stationariy test, e.g., ADF (augmented Dickey-Fuller) for stationarity
                        max_p = cm_params["max_p"],
                        max_q = cm_params["max_q"],
                        seasonal=True, #set to seasonal 
                        start_P=cm_params["start_P"],
                        start_Q=cm_params["start_Q"],
                        max_Q=cm_params["max_Q"],
                        m=cm_params["m"], # freqeuncy of the cycle (i.e., 12 periods ...12 months)
                        D=cm_params["D"], #order of the seasonal differencing ... will be estimated when seasonality = True
                        trace=cm_params["trace"],   # print model AIC 
                        error_action=cm_params["error_action"],  # don't want to know if an order does not work
                        suppress_warnings=cm_params["suppress_warnings"], # don't want convergence warnings
                        stepwise=cm_params["stepwise"]# stepwise search
                        )
                    m.fit(Y_train)  # fit model in place ... a little different than arima and sarimax
                          
     
                else:     # pdarima w/ exenvars
                    if debug == True:
                        print("test_train_predict: auto_arima fit w/ exenvars")
                    Xexen_train = dfXexen_train.values
                    m = pm.auto_arima(Y_train, exogenous = Xexen_train , 
                                    start_q = cm_params["start_q"],
                                    start_p = cm_params["start_p"],
                                    d = cm_params["d"],
                                    test = cm_params["test"], # stationariy test, e.g., ADF (augmented Dickey-Fuller) for stationarity
                                    max_p = cm_params["max_p"],
                                    max_q = cm_params["max_q"],
                                    seasonal=True, #set to seasonal 
                                    start_P=cm_params["start_P"],
                                    start_Q=cm_params["start_Q"],
                                    max_Q=cm_params["max_Q"],
                                    m=cm_params["m"], # freqeuncy of the cycle (i.e., 12 periods ...12 months)
                                    D=cm_params["D"], #order of the seasonal differencing ... will be estimated when seasonality = True
                                    trace=cm_params["trace"],   # print model AIC 
                                    error_action=cm_params["error_action"],  # don't want to know if an order does not work
                                    suppress_warnings=cm_params["suppress_warnings"], # don't want convergence warnings
                                    stepwise=cm_params["stepwise"]# stepwise search
                                    )
                    m.fit(Y_train,exogenous = Xexen_train) # fit model in place w/ exen ... similar to sklearn ... different than arima and sarimax

                    
            if model_type == "sk": # fit sk model
                X_train = dfX_train.values  # last train index is i-1 
                m = {} # by default model is put in a dictionary for each covariate ... will be overwritten with a single model if it is univariate
                for _cv in covars:    # fit model to all covars
                    if debug == True:
                        print(f'test_train_predict: sk fit, model_type = {model_type} i = {i}, _covar = {_cv} ') 
                    dfY_train = dfXY_train[_cv]
                    Y_train = dfY_train.values
                        
                    if horizon_predict_method == "single_step":  # only one model fit for the current X_train ... predict for all horizons in single step fashion
                        
                        # clone model
                        if len(covars) > 1: # multivariate, one model per covar
                            m[_cv] = clone(model[_cv])
                            m[_cv].fit(X_train,Y_train)
                        else :             # univariate, one model
                            m = clone(model)
                            m.fit(X_train,Y_train)
                        
                        if debug == True:
                            print(f'test_train_predict: sk fit, covariate _cv = {_cv}')
                            print(f'test_train_predict: sk model fit, X_train.shape =',X_train.shape)
                            print(f'test_train_predict: sk model fit, Y_train =',Y_train)
                            
                    elif horizon_predict_method == "multi_step": # Nhorizon > 1 for multistep, family of models, one for each time-step within a forecast horizon
                        
                        # fit 0-th model outside the loop since the slicing will not work properly with n = 0
                        n = 0
                        # fit model for n = 0
                        if len(covars) > 1: # multivariate, one model per covar
                            model_n = clone(model[_cv])
                            model_n.fit(X_train,Y_train[n]) 
                            m[_cv] = [model_n]
                        else :             # univariate, one model
                           model_n = clone(model) # reference to the model   
                           model_n.fit(X_train,Y_train[n])   
                           m = [model_n]
                           
                        if debug == True:
                            print(f'test_train_predict: sk model fit, X_train.shape =',X_train.shape)
                            print(f'test_train_predict: sk model fit, Y_train[{n}].shape =',Y_train[n].shape)
                        
                        # fit model for n = 1 to Nhorizon (-1)
                        for n in np.arange(1,Nhorizon,1):
                            _X = X_train[:-n] # can use up to -n from end since there are NaNs in Y
                            _Y_train=[dfY_train[_cv].shift(-n).values] # shift -n ... will contain n NaN s at the end ... clip off below   
                            _Y = Y_train[:-n]  # avoid NaNs at end of Y
                            
                            if len(covars) > 1: # multivariate, one model per covar
                                model_n = clone(model[_cv])
                                model_n.fit(_X,_Y) 
                                m[_cv] = m[_cv] +  [model_n]
                            else :             # univariate, one model
                                model_n = clone(model) # reference to the model   
                                model_n.fit(_X,_Y)   
                                m = m + [model_n]
                        
                            if debug == True:
                                print(f'test_train_predict: sk model fit m[{n}] = {m[n]}')
                                print(f'test_train_predict: sk model fit, X_train.shape =',X_train.shape)
                                print(f'test_train_predict: sk model fit, Y_train[{n}].shape =',Y_train[n].shape)
                                
                        if debug == True:
                            if len(covars) > 1:
                                print("train_test_predict: m.keys = { " ,".join(list(m.keys()))}")
             

            if model_type == "tf": # fit tf model
                if debug == True:
                    print(f'train_test_predict: tf fit i = {i}')
                    
                dfY_train = dfXY_train[covars]
                Y_train = dfY_train.values
                
                if debug == True:
                    print("test_tran_predict: tf fit, all covars dfY_train.tail()")
                    display(dfY_train.tail())
                    print(f'test_train_predict: tf fit, dfY_train.shape = {dfY_train.shape}')

                 # dfXlags 
                 # ... exenvars not yet removed fro dfXlags_train 
                 # 
                dfXlags_train = dfX_train 
                
                
                
                Xlags_train = dfXlags_train.values
                if debug == True:
                    exen_str = "with exenvars" if exenvars !=None else ""
                    print(f'test_train_predict: tf fit, dfXlags_train {exen_str}')
                    display(dfXlags_train.tail(3))
                    print(f'test_train_predict: dfXlags_train.shape = {dfXlags_train.shape} ')
                
                # dfXexnvars lstm == False ... if exenvars list of list and lstm == False remove exenvars from dfXlags
                # leave exenvars with dfXlags if exenvars is list of strings
                # if exenvars list of list then remove exenvars from dfXlags
                if tf_params["lstm"] == False:
                    if exenvars == None: Xexen_pred = None 
                    if (exenvars != None and not isinstance(exenvars[0],str)):   # exenvars list of strings then exenvars are with dfXlags
                        dfXexen_train = dfXlags_train[exenvars_flat]
                        dfXlags_train = dfXlags_train.drop(exenvars_flat,axis=1)
                        Xlags_train = dfXlags_train.values
                        # exenvars is list of lists ... create Xexen_train list of list
                        Xexen_train = []
                        for exen in (exenvars): Xexen_train = Xexen_train + [dfXexen_train[exen].values]
                        if debug == True:
                            print(f'train_test_predict: exenvars list of lists, Xexen_train')
                            print(Xexen_train)
                
                
                # dfXexnvars lstm == True
                if tf_params["lstm"] == True:
                    if (exenvars != None):
                        dfXexen_train = dfXlags_train[exenvars_flat]
                        dfXlags_train = dfXlags_train.drop(exenvars_flat,axis=1)
                        Xlags_train = dfXlags_train.values
                        
                        if not isinstance(exenvars[0],str):
                            Xexen_train = []
                            for exen in (exenvars): 
                                print("test_train_predict, exen =", exen)
                                Xexen_train = Xexen_train + [dfXexen_train[exen].values]
                        else:
                            Xexen_train = dfXexen_train.values
                        if debug == True:
                            print(f'train_test_predict: exenvars Xexen_train')
                            print(Xexen_train)
                    else:
                        Xexen_pred = None
                            
                if debug == True:
                    print(f'test_train_predict: tf fit, dfXlags_train')
                    display(dfXlags_train.tail(3))

                # X cats is a list of dataframes ... embeddings will be a list of inputs
                Xcats_train_list =  [dfXcats_train[c].values for c in catvars] if catvars != None else None
                
                if debug == True:
                    print(f'train_test_predict: Xcats_train_list')
                    print(Xcats_train_list)

                # X_train
                X_train =  [ Xlags_train.reshape(Xlags_train.shape[0], Nlags, Ncovars) ] if tf_params["lstm"] == True else [Xlags_train]
                if exenvars != None and catvars != None: 
                    if not isinstance(exenvars[0],str):
                        if Nlags > 0:
                            for Xexen_t in Xexen_train: X_train = X_train + [Xexen_t]
                        else:
                            X_train =  []
                            for Xexen_t in Xexen_train: X_train = X_train + [Xexen_t]
                        X_train = X_train + Xcats_train_list if catvars != None else X_train

                    else: # exenvars is list of strings
                        if tf_params["lstm"] == True: 
                            if Nlags > 0:
                                X_train = X_train + [Xexen_train] +  Xcats_train_list  if catvars != None else X_train 
                            else:
                                X_train = [Xexen_train] +  Xcats_train_list  if catvars != None else X_train 
                            
                        else: # exenvars inside of X_train
                            X_train = X_train +  Xcats_train_list
                elif exenvars == None and catvars != None:
                    X_train = X_train + Xcats_train_list          
                else:
                    X_train = X_train # this is the default ... exenvars == None catvars == None
                
                if debug == True:
                    for n in np.arange(0,len(X_train)):
                        print(f'test_train_predict: X_train[{n}].shape = {X_train[n].shape}')
                        print(f'X_train[{n}][{X_train[n].shape[0]-1}] = {X_train[n][X_train[n].shape[0]-1]}')
   
                if i == i_initial_pred:
                    if verbose == True or debug == True:
                        print(f'test_train_predict: tf initial fit, i = {i}')
                        print("Nepochs_i =",tf_params["Nepochs_i"])
                        #print(f'test_train_predict: X_train = {X_train}')
                    
                    m=model # set a pointer to the model ... do not clone ...cloning will require compiling and implementng redundent TF logic ... dont do that outside of sforecast
                                        
                    history_i=m.fit(X_train,Y_train,epochs=tf_params["Nepochs_i"], batch_size = tf_params["batch_size"], verbose=0)
                else:
                    # this will tune the previously trained model ... Nepochs_t < Nepochs_i
                    
                    if verbose == True:  
                        print(f'test_train_predict: tf fit/tune, i = {i}')
                        print("Nepochs_t =",tf_params["Nepochs_t"])
                    
                    
                    history_t=m.fit(X_train,Y_train,epochs=tf_params["Nepochs_t"], batch_size =tf_params["batch_size"], verbose=0)      

        
        ##### test-predict , predict ####
        # predict if predict==True and i <=i_last_predict
        # predict if fit==True ... and i <=i_last_predict and if Ntest > 0  
        
        #### Inner Loop .. test, predict ...
        inner_loop_break = False
        for k in np.arange(0,N_loop_step,1):
            if inner_loop_break == True:
                break
            if debug == True:
                print("  train_test_predict: inner-loop, test-predict, predict, i <= i_last_predict, i_last_predict = ",i_last_predict, ", i = ",i, ",k =",k) 
            ## X_test ... Xexen_test, Xcats_test
            if i <= i_last_predict:  #  the last row is only for the final fit ... if this is i_lastrow the prediction happened on previous i ...
                
                if (fit==True and  Ntest > 0) or predict==True:   #### fit-test/predict or predict ####
                    
                    ##########################
                    #### X_test/pred  Xexen_test/red  
                    if model_type == "sk":   # sk X_test and predict
                        
                        # N-step forecast with k inner loop
                        if debug == True:
                            print(f'  test_train_predict: sk, test, predict, {horizon_predict_method}  i =  {i} , k = {k}')                        
                        
                        if horizon_predict_method == "single_step" or k == 0:   # single_step any k or multi_step only when k == 0
                        
                            p_index = [dfxy.index[i+k]] if fit == True else [index_pred[i+k]]
                            
                            dfXY_pred = pd.DataFrame(data = {_cv:np.NaN for _cv in covars } , columns = covars, index = p_index)
                            
                            # update y with previous prediction in covarlags
                            if (fit == True and k > 0) or (predict == True and predict_cnt > 0): 
                                # update the previous y in make_lags
                                for _cv in covars:
                                    make_lags.set_last_y(_cv,y_pred_nda[_cv][predict_cnt-1], debug=debug)
                            # lags
                            dfXY_pred = make_lags.transform(dfnewrows=dfXY_pred, Nout = 1, debug=debug)
                            
                            
                            dfX_pred = dfXY_pred.drop(covars,axis=1)
                            
                                                            
                            # exogs
                            #dfXexen[exogvars] = dfxy.iloc[i+k:i+k+1][exogvars].values if exogvars != None else dfXexen
                            if fit == True:
                                if exogvars != None: dfX_pred[exogvars] = dfxy.iloc[i+k:i+k+1][exogvars].values
                            elif predict == True:
                                if exogvars != None: dfX_pred[exogvars] = pred_params["dfexogs"].iloc[i+k:i+k+1].values
                            
                            # derived attributes
                            if derived:
                                dfX_pred = make_derived_attributes.transform(dfnewrows=dfX_pred,Nout=1 )
                             
                                
                            # scale
                            dfX_pred = scaler.transform(dfX_pred)
                            X_pred = dfX_pred.values

                                    
                        if debug == True:
                            print(f'  test_train_predict: sk test, predict {horizon_predict_method}, dfX_pred.tail()')
                            display(dfX_pred.tail(3))

                    
                    elif model_type == "tf": # tf X_test/pred  ... only single-step method (1-step k=1:Nhorizon) is supported
  
                        # dfXY_pred
                        p_index = [dfxy.index[i+k]] if fit == True else [index_pred[i+k]]
                        dfXY_pred = pd.DataFrame(data = {_cv:np.NaN for _cv in covars } , columns = covars, index = p_index)
                        
                        #Lags
                        # update y with previous prediction in covarlags
                        if (fit == True and k > 0) or (predict == True and predict_cnt > 0): # 
                            # update the previous y in make_lags
                            for _cv in covars:
                                make_lags.set_last_y(_cv,y_pred_nda[_cv][predict_cnt-1])  #
                                    
                        dfXY_pred = make_lags.transform(dfnewrows=dfXY_pred, Nout = 1, debug=debug)
                        dfX_pred = dfXY_pred.drop(covars,axis=1) 
                        
                        # exogs
                        #dfXexen[exogvars] = dfxy.iloc[i+k:i+k+1][exogvars].values if exogvars != None else dfXexen
                        if fit == True:
                            if exogvars != None: dfX_pred[exogvars] = dfxy.iloc[i+k:i+k+1][exogvars].values
                        elif predict == True:
                            if exogvars != None: dfX_pred[exogvars] = pred_params["dfexogs"].iloc[i+k:i+k+1].values
                            
                        
                        # endogs ... derived 
                        if derived:
                            dfX_pred = make_derived_attributes.transform(dfnewrows=dfX_pred,Nout=1 )
                            
                        if debug == True:
                            exen_str = "w exenvars" if exenvars != None else ""
                            print(f'  test_train_predict: df_pred {exen_str}')
                            display(dfX_pred.tail(3))

                        # scale
                        dfX_pred =  scaler.transform(dfX_pred) # scaler.transform(dfX_pred.drop(catvars,axis=1)) if catvars != None else
                        
                        #Xlags_pred  
                        dfXlags_pred = dfX_pred # may contain exenvars if exenvars != None
                        if debug == True:
                            exen_str = "with exenvvars"  if exenvars != None else ""
                            print(f'test_train_predict: dfXlags_pred {exen_str}')
                            display(dfXlags_pred.tail(3))
                            
                        # dfXexen lstm == False
                        #  create Xexen_pred
                        #  remove exenvars from dfXlags
                        if tf_params["lstm"] == False:
                            if exenvars == None: Xexen_pred = None
                            if exenvars != None and not isinstance(exenvars[0],str): # list of lists
                                dfXexen_pred = dfX_pred[exenvars_flat]
                                dfXlags_pred = dfXlags_pred.drop(exenvars_flat, axis =1) 
                                Xexen_pred = []
                                for exen in (exenvars): Xexen_pred = Xexen_pred + [dfXexen_pred[exen].values]
                                if debug == True:
                                    print(f'test_train_predict: dfXexen_pred')
                                    display(dfXexen_pred.tail(3))
                                    print(f'train_test_predict: exenvars list of lists, Xexen_pred')
                                    print(Xexen_pred)
                         
                        # dfXexnvars lstm == True    
                        if tf_params["lstm"]== True:
                            if exenvars == None: Xexen_pred = None
                            if exenvars != None: # list of lists
                                dfXexen_pred = dfX_pred[exenvars_flat]
                                dfXlags_pred = dfXlags_pred.drop(exenvars_flat, axis =1) 
                                if not isinstance(exenvars[0],str):
                                    Xexen_pred = []
                                    for exen in (exenvars): Xexen_pred = Xexen_pred + [dfXexen_pred[exen].values]
                                else:
                                    Xexen_pred = dfXexen_pred.values

                        Xlags_pred = dfXlags_pred.values
                        if debug == True:
                            print(f'test_train_predict: dfXlags_pred')
                            display(dfXlags_pred.tail(3))
                                
                                
                        #Xcats_predLlist
                        if catvars != None:
                            if fit ==True:
                                dfXcats_pred = dfxy[catvars].iloc[i+k:i+k+1]
                            elif predict == True:
                                dfXcats_pred = pred_params["dfcats"].iloc[i+k:i+k+1]
                            Xcats_pred_list =  [dfXcats_pred[c].values for c in catvars] 
                            if debug == True:
                                print(f'  test_train_predict: dfXcats_pred')
                                display(dfXcats_pred.tail())
                                
                        
                        #[ Xlags_train.reshape(Xlags_train.shape[0], Nlags, Ncovars) ]
                        # X_pred 
                        X_pred = [ Xlags_pred.reshape(1, Nlags, Ncovars)] if tf_params["lstm"] == True else [Xlags_pred] # Xlags_pred.shape[0]
                        if exenvars != None:
                            if not isinstance(exenvars[0],str):
                                if Nlags > 0:
                                    for Xexen_p in Xexen_pred:  X_pred = X_pred + [Xexen_p] 
                                else:
                                    X_pred = []
                                    for Xexen_p in Xexen_pred:  X_pred = X_pred + [Xexen_p] 
                                    
                                X_pred = X_pred +  Xcats_pred_list  if catvars != None else X_pred
                            else:  
                                if tf_params["lstm"] == True:
                                    if Nlags >0:
                                        X_pred = X_pred +[ Xexen_pred ] + Xcats_pred_list  if catvars != None else X_pred
                                    else:
                                        X_pred = [ Xexen_pred ] + Xcats_pred_list  if catvars != None else X_pred
                                else:
                                    X_pred = X_pred + Xcats_pred_list  if catvars != None else X_pred
                        elif exenvars == None and catvars != None:
                            X_pred = [Xlags_pred] + Xcats_pred_list
                        else:
                            X_pred = Xlags_pred
                            
                        if debug == True:
                            print(f'  test_train_predict: X_pred, i = {i}, k = {k}') 
                            for n in np.arange(0,len(X_pred)):
                                print(f'  test_train_predict: X_pred[{n}].shape {X_pred[n].shape} ')  
                            print(f'  test_train_predict: X_pred = {X_pred}')  
                            
                            
                    elif model_type == "cm": # Xexen_test, Xexog_test 
                        if fit == True and exogvars != None and not derived:
                            #Xexogs_test = dfX_train[exogvars][i+k :i+k+1].values
                            
                            # Nhorizon = 1 for arima and sarimax ... in the future could emulate a multi-step forecast
                            # Nhorizon could be > 1 for auto_arima and not derived
                            if cm_params == "auto_arima":
                                dfXexogs_pred = dfxy[exogvars][i:i+ Nhorizon]
                                dfXexen_pred = dfXexogs_pred
                            else:
                                dfXexogs_pred = dfxy[exogvars][i + k:i+ k + 1]
                                dfXexen_pred = dfXexogs_pred    
                            
                            if debug == True:
                                print("  test_train_predict: dfXexen_test = dfXexogs_test ")
                                display(dfXexen_pred)
    
                            Xexen_pred = dfXexogs_pred.values
                            
                        if predict == True and exogvars != None and not derived: # predict ... exogvars not derived
                            dfXexogs_pred = pred_params["dfexogs"]
                            if debug == True:
                                print("  test_train_predict: predict, dfXexogs=")
                                display(dfXexogs.tail())
                                print("  test_train_predict: predict, index_pred =", index_pred)
                                print("  test_train_predict: predict, k = ", k)
                            
                            Xexen_pred = dfXexogs_pred.values
    

                        if derived:   #  fit and derived, therefore lags needed ... exogvars may or may not be needed
                            #X_test = dfX.iloc[i:i+1].values # get the prediction row
                            #if exenvars != None:

                            #print("train_test_predict: cm fit Xexen_test, i = ",i)
                            # prev y or yhat
                            # y_pred_nda
                            # pevious y ... will become y_m1 with make_lags
                            if debug == True:
                                print("  test_train_predict: fit k = ", k)
                                print("  test_train_predict: y =", y)
                            # if k == 0 then prev_y  else if k > 0 then prev_y = prev prediction
                            
                            # do not yet have a y value (unlagged) so set the current y to np.NaN 
                            # ... update later when have a predicted value ... need this to create lags
                            
                            p_index = [dfxy.index[i+k]] if fit == True else [index_pred[k]]
                            dfXexen_pred = pd.DataFrame(data = [np.NaN] , columns = y, index = p_index )

                            # update y with previous prediction in covarlags when k > 0
                            if k > 0: 
                                # update the previous y in make_lags
                                make_lags.set_last_y(y,y_pred_nda[k-1], debug=True)
                                if debug == True:
                                    print("  test_train_predict: cm k =", k , ",y =", y)
                                    print("  test_train_predict: cm y_pred_nda[k-1] =",y_pred_nda[k-1])
                                # only autoarima with derived variables will use this .. y_pred_nda[k-1] is a scaler
                                
                            # exogs
                            if exogvars != None: 
                                dfXexen_pred[exogvars] = dfxy.iloc[i+k:i+k+1][exogvars].values if fit == True else pred_params["dfexogs"].iloc[k:k+1].values 

                            # make_lags
                            dfXexen_pred = make_lags.transform(dfnewrows=dfXexen_pred, Nout = 1, debug=debug)
                            dfXexen_pred = dfXexen_pred.drop(y,axis=1)
                            
                            # derived attributes
                            dfXexen_pred = make_derived_attributes.transform(dfnewrows=dfXexen_pred,Nout=1 )
                            dfXexen_pred = dfXexen_pred[exenvars]
                            
                            if debug == True:
                                print("  test_train_predict: cm fit predict dfXexen_test w derived =")
                                display(dfXexen_pred)
                            
                            Xexen_pred = dfXexen_pred.values
                            
                            
                    #### test-predict, predict  #### 
                    
                    if model_type == "cm": # predict
                        
                        if  cm_params["model"] == "arima":
                                if debug == True:
                                    print ("  train_test_predict: cm fit predict i = ",i)
                                y_pred = model_fit.forecast().reshape(1,1)  # arima returns a different format than sarimax below

                        if  cm_params["model"] == "sarimax" :
                            
                            if not derived:  #
                                if exogvars == None:
                                    y_pred = model_fit.forecast()[0]
                                else:
                                    y_pred = model_fit.forecast(exog=Xexen_pred)[0]
    
                            else: # derived variables
                                if derived == True:
                                    if debug == True:
                                        print("  train_test_fit: cm, predict with exen")                              
                                        print("  train_test_predict: cm fit, Xexen =",Xexen_pred)
                                    
                                    y_pred = model_fit.forecast(exog=Xexen_pred)[0]
                                
                            y_pred = np.array([y_pred]).reshape(1,1)
                            
                            if debug == True:
                                print("test_train_predict, cm, sarimax, y_pred =",y_pred)
                            
                        
                        if cm_params["model"]== "auto_arima": # predict
                                
                            #call forecast on the fitted model m
                            #n_periods = X_test.shape[0]
                            #print("n_periods = ",n_periods)
                            if debug == True:
                                print("test_train_predict: autoarima, model m =",m)

                            if exenvars == None:
                                n_periods = autoarima_nperiods  # N_loop_step = Nhorizon ... 
                                y_pred = m.predict(n_periods) 
                                # reshape for consistenty with y_test, eg ... error = y_test - y_pred 
                                y_pred = np.array([y_pred]).reshape(n_periods,1) 
                                inner_loop_break = True # complete the inner loop since did an nperiod forecast/predictin
                                if debug == True:
                                    print("  test_train_predict: ypred autoarima  = ", y_pred)
                                    print("  test_train_predict: ypred autoarima, n_periods = ", n_periods)
                                    print("  test_train_predict: auto_arima, inner_loop_break = ",inner_loop_break)
                                
                            elif not derived and exogvars != None:
                                n_periods = autoarima_nperiods  # N_loop_step = Nhorizon ... 
                                y_pred = m.predict(n_periods,exogenous = Xexen_pred) 
                                # reshape for consistenty with y_test, eg ... error = y_test - y_pred 
                                y_pred = np.array([y_pred]).reshape(n_periods,1) 
                                inner_loop_break = True # complete the inner loop since did an nperiod forecast/predictin
                                if debug == True:
                                    print("  test_train_predict: ypred autoarima w exogs and not derived = ", y_pred)
                                    print("  test_train_predict: ypred autoarima,  n_periods = ", n_periods)
                                    print("  test_train_predict: auto_arima, inner_loop_break = ",inner_loop_break)
                                
                            else:  # exenvars not None
                                y_pred = m.predict(n_periods = 1, exogenous = Xexen_pred)
                                if debug == True:
                                    y_pred = np.array([y_pred]).reshape(1,1)
                                
                            if debug == True:
                                print("  test_train_predict: autoarima")
                                print("  test_train_predict, cm, y_pred =",y_pred)
                            
                    elif model_type == "sk": # sklearn predict
                        y_pred = {}
                        if horizon_predict_method == "single_step":
                            for _cv in covars: 
                                if len(covars) == 1: # univariate
                                    y_pred[_cv] = m.predict(X_pred) # X_pred changes for each k, model stays constant
                                else: # multi-variate
                                    y_pred[_cv] = m[_cv].predict(X_pred) # X_pred changes for each k, model stays constant
                                if debug == True:
                                    print(f'  test_train_predict: sk test, predict, signle_step, model_type = {model_type}, m, ypred[{_cv}] = {y_pred[_cv]}')
                        
                        elif horizon_predict_method == "multi_step":
                            if len(covars) > 1:
                                y_pred = m[_cv][k].predict(X_pred) # prediction ... X_pred changes only with k == 0, model changes for each k
                            else:
                                y_pred = m[k].predict(X_pred) 
                            if debug == True:
                                print(f'  test_train_predict: sk test, predict, multi_step, model_type = {model_type}, m[{k}], ypred = {y_pred}')
                        
                        if debug == True:
                            print(f'test_train_predict: y_pred.keys() = {", ".join(list(y_pred.keys()))}  ') 
                            
                    else: # tf predict ... must, predict all covariates simultaneously
                        
                        y_pred = m.predict(X_pred)
                        
                        if debug== True:
                            print(f'  test_train_predict: tf predict, i ={i}, k = {k}')
                            print(f'  test_train_predict: tf predict, y_pred = {y_pred}, y_pred.shape = {y_pred.shape}')
                            print(f'  test_train_predict: tf predict, y_pred.shape = {y_pred.shape}')
                        
                ### end of predict ... assimalate pred results y_pred_nda, y_test_nda, y_pred_idx
                
                    if debug == True:
                        print("test_train_predict: predict_cnt =",predict_cnt) 
                        
                    ### y_pred_nda
                    if model_type == "sk": # y_pred_nda and y_test_nda
                        
                        if fit == True:
                            dfcv_test = dfxy[covars].iloc[i + k:i + k + 1]
                            y_test = dfcv_test.values
                        
                            if debug == True:
                                print(f'  test_train_predict: k dfcv_test, i = {i}, k = {k}')
                                display(dfxy[covars])
                        
                        if i == i_initial_pred and k == 0:
                            y_pred_nda = {}
                            if fit == True: y_test_nda = {}

                        for n,_cv in enumerate(covars): # y_pred and  y_test for each covar
                            if (i != i_initial_pred or k > 0 ):
                                y_pred_nda[_cv] = np.append(y_pred_nda[_cv],y_pred[_cv],axis=0) 
                                if fit == True: y_test_nda[_cv] = np.append(y_test_nda[_cv],np.array(y_test[0][n]).reshape(1,1),axis=0) 
                            else:
                                y_pred_nda[_cv] = y_pred[_cv]
                                if fit == True: y_test_nda[_cv] = np.array(y_test[0][n]).reshape(1,1)
                                   
                        if fit == True:
                            _pred_idx = [i + k + Nclip ]  # iloc index (ith)
                            y_pred_idx = _pred_idx if (i == i_initial_pred and k == 0 ) else y_pred_idx + _pred_idx                       
                    
                    # save indexes to put back into original data frame ... adjust for Nclip ... externally there is no clip so add Nclip back to get the -th index correct                      
                    elif model_type == "cm" and cm_params["model"] == "auto_arima" and not derived:

                            if fit == True:
                                y_test = dfy.iloc[i:i + Nhorizon].values 
                                #    _n_idx takes care of potential idx mismatch for such a case
                                _n_idx = y_test.shape[0] # this is iloc type index not loc index
                                _pred_idx = list(np.arange(i,i + _n_idx,1)+ Nclip)
                                y_pred_idx = _pred_idx if (i == i_initial_pred and k == 0 ) else y_pred_idx + _pred_idx
                                y_test_nda = np.append(y_test_nda,y_test,axis=0) if (i != i_initial_pred or k > 0 ) else y_test 

                            y_pred_nda = np.append(y_pred_nda,y_pred,axis=0) if (i != i_initial_pred or k > 0 ) else y_pred 
                            
                            if predict == True:
                                outer_loop_break = True
                                
                    elif model_type == "cm":  # cm autoarima with derived/endogenous
                        
                        y_pred_nda = np.append(y_pred_nda,y_pred,axis=0) if (i != i_initial_pred or k > 0 ) else y_pred 
                        _pred_idx = [i + k + Nclip ]  # iloc index (ith)
                        y_pred_idx = _pred_idx if (i == i_initial_pred and k == 0 ) else y_pred_idx + _pred_idx
                        
                        if fit == True:
                            y_test = dfy.iloc[i + k:i + k + 1].values
                            y_test_nda = np.append(y_test_nda,y_test,axis=0) if (i != i_initial_pred or k > 0 ) else y_test
                                
                    elif model_type == "tf": # tf ... assimilate pred results y_pred_nda, y_test_nda
                    
                        if fit == True:
                            dfcv_test = dfxy[covars].iloc[i + k:i + k + 1]
                            y_test = dfcv_test.values
                            if debug == True:
                                print(f'  test_train_predict:dfcv_test, dfcv_test.shape = {dfcv_test.shape}, i = {i}, k = {k}')
                                display(dfcv_test)
                            
                        if i == i_initial_pred and k == 0:
                            y_pred_nda = {}
                            if fit == True: y_test_nda = {}
                                                
                        for n,_cv in enumerate(covars): # y_pred and  y_tests for each covar
                            if (i != i_initial_pred or k > 0 ):
                                y_pred_nda[_cv] = np.append(y_pred_nda[_cv],np.array(y_pred[0,n]).reshape(1,1),axis=0) 
                                if fit == True: 
                                    y_test_nda[_cv] = np.append(y_test_nda[_cv],np.array(y_test[0][n]).reshape(1,1),axis=0)
                            else:
                                # first time assert that prediction is the same length as covars ... test occurs len(covar) times
                                assert len(y_pred[0]) == len(covars), f'len(y_pred) = len(covars), predicted outputs must equal the number covariates.'
                                y_pred_nda[_cv] = np.array(y_pred[0,n]).reshape(1,1)
                                if fit == True: y_test_nda[_cv] = np.array(y_test[0][n]).reshape(1,1)
                            
                        if debug == True:
                            print(f'  test_train_predict: tf y_pred_nda = {y_pred_nda}')
                            if fit == True:
                                print(f'  test_train_predict: tf y_test_nda = {y_test_nda}')
                        
                        if fit == True:
                            _pred_idx = [i + k + Nclip ]  # iloc index (ith)
                            y_pred_idx = _pred_idx if (i == i_initial_pred and k == 0 ) else y_pred_idx + _pred_idx    
                        
                    if predict == True:
                        _pred_idx = [index_pred[predict_cnt]]
                        y_pred_idx = y_pred_idx + _pred_idx if (predict_cnt > 0) else  _pred_idx
                        y_test = None
                        y_test_nda = None   

                    predict_cnt += 1
                                
                else: # fit = True and Ntest = 0
                # if not predicting then set test and pred variables to NaN
                    y_test_nda = None
                    y_pred_nda = None
                    #y_pred_idx = dfX.index[dfX.index.size - 1 ]
                    
                if debug == True:
                    print(f' test_train_predict:  i = {i}, k = {k}, i_initial_pred = {i_initial_pred}')
                    print("  test_train_predict: y_pred_nda =",y_pred_nda)
                    print("  test_train_predict: y_test_nda =",y_test_nda)
                    print("  test_train_predict: _pred_idx", _pred_idx)
                    print("  test_train_predict: y_pred_idx",y_pred_idx)
                
    return_tuple = None 
    i_initial_pred = i_initial_pred + Nclip # adjust for the next level up, where Nclip was not removed from nitial rows
    
    if model_type == "cm":
        return_tuple = y_pred_nda, y_test_nda, y_pred_idx, dfXY_train, m, model_fit, i_initial_pred, make_derived_attributes, make_lags, scaler
    if model_type == "sk":
        return_tuple = y_pred_nda, y_test_nda, y_pred_idx, dfXY_train, m, i_initial_pred, make_derived_attributes, make_lags, scaler
    elif model_type =="tf":
        return_tuple = y_pred_nda, y_test_nda, y_pred_idx, dfXY_train, m, history_i, history_t,i_initial_pred, make_derived_attributes, make_lags, scaler
            
    return return_tuple  ### test_train_predict ###


def sliding_forecast(df, y, model, model_type="sk", swin_params=None, cm_params=None, tf_params=None, scale_params = None,
                     fit=True, predict=False, predict_params = None,  verbose = False, debug = False):
    
    '''Sliding_forecast() manages the fit and predict operations. It calls test_train_predict() function and receives the 
    resulting forecast/predictions, and indexes (corresponding to the input dataframe). Sliding_forecast() creates the return data frame, 
    manages the creation of the return parameters and objects.
    
    When called with the fit = True (called by sforcast.fit), sliding_forecast() manages
    the creation of metrics including confidence intervals and fit metrics such as RMSE, and MAE. Sliding_foreast() will return 
    out-of-sample train/test predictions, the fitted model (recursive fit/prediction) or models (direct fit/prediction), fitted scaler transform, 
    fitted dervived variables transform. Accuracy metrics are formulated based on each train-test window.
    
    When called with input predict = True (called by sforecast.predict) sliding_forecast()
    passes the fitted objects (model, make_lags, scaler, make_derived_attributes) to test_train_predict(). Subsequently, sliding_foreast receives the predictions and 
    creates the return DataFrame of predictions.
    
    Args:
    
        df(DataFrame): DataFrame containing x and y (target) variables. When called with fit = True, dfX is passed to test_train_predict. This DataFrame is ignored during the predict operation. The DataFrame is expected to be sorted in timeseries ascending time order, and may have an integer or time index.
        
        model (ml model): An ML model either from SKLearn, TensorFlow, or classical forecastng model. If fit = True the model is trained or if predict == True, the trained model is used for foreward predcitions.
        
        model_type (str): Default = "sk" or "tf". Default = "sk".
        
        swin_params (dictionary): refer to sforecast swin_params
        
        tf_params (Dictionary, optional): refer to sforecast tf_params
                    
        scale_params (Dictionary, optional): refer to sforecast scale_params
        
        fit (Booelan): Default = True. Fit the ML model, scaler, make_lags transform, and derivived attributes_transform with out-of-sample (train/test) observations. Eiither fit = True or predict = True is required.
        
        predict (Boolean): Default = False. When predict = True use the trained (fitted) model and objects (make_lags, scaler, derived_attributes transform) for predictions. Variables will be scaled with the fitted scalers.
           
        predict_params (Dictionary, optional). When predict = True the predict_params dictionary contains parameters and objects needed for prediction including make_lags transform, dfexogs (exogenous dataframe), dfxcats (numerical categorical features), and derived attributes transform.
        
        verbose: True or False. Defaults to False.

    Returns:
        when fit = True 
            * dfXY: XY forecast DataFrame including lagged variables. The rows include all observations. The last row returned corresponds the last training row dfX (after removing y) for predicting/forecast the next N-step forecast (N-step = Nhorizon)
            * df_pred: predictions. See documentation for sforcast.forecast.
            * metrics: Dictionary containg MSE and MAE for the corresponding predictions
            * m (model): ML forecast model. When model_type = "sk" or "tf" and "cm" AUTO_ARIMA , m corresponds to the fitted model, after the last traiing.
            * history_i: TensorFlow training history (initial training)
            * history_t: TensorFlow training (tunning) history corresponding to the final (last) training.
            * m_fit: Fitted model for the case of model_type = "cm" ARIMA and ARIMAX , otherwise = None.
            * scaler: fitted scaler transform of MinMax, StandardScaler or mixed scaling as specified by the scaler_params
            * make_derived_attributes: fitted derived attributes (endogenous) variables transform
            * make_lags: fitted lagged covariates transform
            * ci: confidence intervals
            
        when predict = True 
            * df_pred: dataframe of predictions
        
   '''

    # extract variables from input dictionaries
    minmax = swin_params["minmax"]
    Ntest = swin_params["Ntest"]  
    Nhorizon = swin_params["Nhorizon"] 
    alpha = swin_params["alpha"]
    ci_method = swin_params["ci_method"]
    covars = swin_params["covars"]
    
    ci = {} # initialize empty confidence interval dictionary
    
    if debug == True:
        print(f'sliding_forecast: y = {y}')
     
    # setup m ... return model
    if len(y) > 1 and ( model_type == "sk" or model_type == "cm"):
        m = {}
        m_fit = {}
    else:
        m = None
        m_fit = None

    if debug == True:
        print(f'sliding_forecast: {model_type}, model  = {type(model)}')
        print("sliding_forecast: type(m) = ",type(m))
        if model_type == "cm" or model_type == "sk":
            dummy = print(f'sliding_forecast: {model_type}, model keys = {" ,".join(list(model.keys()))}') if len(y )> 1 else None

    #### Assert Fit or Predict not both ####
    # make sure that only one of Fit or Predict is True
    if fit ==True:
        assert predict == False, f'predict must be False if fit == True'
    assert (fit == True or predict == True), f'Either fit or predict must be True'
    
    # predict parameters
    # if predict == False then fit operation will set the predict parameters
    if predict == True:
        index_pred = predict_params["index"] 
        Nhorizon = swin_params["Nhorizon"] 
        make_lags = predict_params["make_lags"]
        make_derived_attributes = predict_params["make_derived_attributes"]
        
    # if fit = True then fit it will manage the predict parameters
    if fit == True:
        predict_params = {}
    
    #### copy df
    # copy the input dataframe to ensure the original is not modified
    #dfXY = None 
    dfXY = df.copy() if fit == True else None # dfXY = None if predict == True


    ###############################    
    ##### Train, Test, Predict  ###
    metrics={}
    df_pred=pd.DataFrame()
    
    # tf traning history variables
    history_i=None #  initial model history (first fit) ... tensorflow
    history_t=None # tune model history (second, third, ... fit) ... tensorflow
    
    # model fit variable
    model_fit=None
    
    # Nh Nhorizon variable
    Nh = Nhorizon 
    
    # TensorFlow            
    if model_type == "tf":
        #### Nhorizon predict forward Nhorizon
        # pred all targets (all _y in y) at once
        # TensorFlow uses one model to make all covariate predictions simultaneously
        
        if debug == True:
            print(f'sliding_forecast: tf,  covars = {covars}')
            print(f'sliding_forecast: tf,  y = {y}')
            print("sliding_forecast: tf, type(model_in) =",type(model))

        y_pred_values, y_test_values, y_pred_idx, dfXY_train, model_out, history_i, history_t, i_initial_pred, make_derived_attributes, make_lags, scaler = train_test_predict(dfXY, y, model, 
                    model_type=model_type, swin_params=swin_params,tf_params=tf_params,scale_params = scale_params,
                    fit=fit, predict=predict, pred_params=predict_params, verbose=verbose, debug=debug)
        
        m = model_out  # TF will return a single model
        
        if fit==True and y_test_values != None:
            for _cv in covars: y_test_values[_cv] = y_test_values[_cv].flatten()
            
        for _cv in covars:
            if (fit == True and Ntest > 0) or predict == True:
                y_pred_values[_cv] = y_pred_values[_cv].flatten()

        if debug == True:
            print("sliding_forecast: model_type = tf, y_pred_values =",y_pred_values)
            print("sliding_forecast: model_type = tf, y_test_values =",y_test_values)
            print("sliding_forecast: model_type = tf , y_pred_idx =",y_pred_idx)

    
    if model_type == "sk":
        # pred all targets (all _y in y) at once
        # one model per y variable ... one model per Nhorizon timestep  
        # if one model and Nhorizon = 1 then model_in is only one model (not a dictionary and not a list)
            
        if debug == True:
            print(f'sliding_forecast: sk,  covars = {covars}')
            print(f'sliding_forecast: sk,  y = {y}')
            print("sliding_forecast: sk, type(model_in) =",type(model))
                    
        y_pred_values, y_test_values, y_pred_idx, dfXY_train, model_out, i_initial_pred, make_derived_attributes, make_lags, scaler  = train_test_predict(dfXY, y, model, 
        model_type=model_type, swin_params=swin_params, cm_params=cm_params, tf_params=tf_params, scale_params = scale_params,
            fit = fit, predict=predict, pred_params=predict_params,verbose=verbose, debug=debug)
             
        m = model_out
        
        if fit==True and y_test_values != None:
            for _cv in covars: y_test_values[_cv] = y_test_values[_cv].flatten()
        
        if debug == True:
            print("sliding_forecast: model_type = sk, y_pred_values =",y_pred_values)
            print("sliding_forecast: model_type = sk, y_test_values =",y_test_values)
            print("sliding_forecast: model_type = sk, y_pred_idx =",y_pred_idx)
            if len(covars) > 1:
                print(f'slidng_forecast: m.keys() = {list(m.keys())}')
            else:
               print(f'slidng_forecast: m = {m}')

    # loop throuh each variable
    # for sk and tf train_test_predict is called above
    # for cm models call train_test_predict for each variable (do not support multivariate forecasting)
    # assimilate prediciton resutls ... for each y, metrics, confidence intervals
    for n,_y in enumerate(y):  

                 
        # classic forecast models arima and sarimax (fyi - see above for auto arima)
        # support only 1-step (Nhorizon = 1), univariate predictition (for every _y)
        if model_type == "cm":

            # Nh = Nhorizon if fit == True or Nhorizon if predict == True
            
            if debug == True:
                print(f'sliding_forecast: _y = {_y}')
                print(f'sliding_forecast: type(model) = {type(model)}')
                
            model_in = model if len(y) == 1 else model[_y] # dict of models, one for every _y
            
            _y_pred_values, _y_test_values, y_pred_idx, dfXY_train, model_out, model_fit, i_initial_pred, make_derived_attributes, make_lags, scaler = train_test_predict(dfXY, _y, model_in, 
                    model_type=model_type, swin_params = swin_params, tf_params=tf_params, scale_params = scale_params,
                    cm_params=cm_params, fit = fit, predict=predict,
                    pred_params=predict_params, verbose=verbose, debug=debug )

            if len(y) == 1:
                m = model_out
                m_fit = model_fit
            else:
                m[_y] = model_out
                m_fit[_y] = model_fit
                
            
            _y_test_values = _y_test_values.flatten() if isinstance(_y_test_values,np.ndarray) else _y_test_values 
            _y_pred_values = _y_pred_values.flatten() if isinstance(_y_pred_values,np.ndarray) else _y_pred_values 
            
            if debug == True:
                print(f'sliding_forecast: _y = {_y}')
                print("sliding_forecast: model_type = cm, _y_pred_values =",_y_pred_values)
                print("sliding_forecast: model_type = cm, _y_test_values =",_y_test_values)
                print("sliding_forecast: model_type = cm, y_pred_idx =",y_pred_idx)
                
    
        elif model_type == "sk":
            _y_pred_values = y_pred_values[_y] if y_pred_values != None else np.NaN
            _y_test_values = y_test_values[_y] if y_test_values != None else np.NaN

        # tf prediction done for all y's simultaneosly above (outside of loop) ... manage the variables inside loop  (here)  
        elif model_type == "tf":
            _y_pred_values = y_pred_values[_y] if y_pred_values != None else np.NaN 
            _y_test_values = y_test_values[_y] if y_test_values != None else np.NaN
            
        # all fits and predictions are done above ... now manage variables
        
        ypred_col = _y +"_pred"
        
        if isinstance(_y_pred_values, np.ndarray):
            # y_pred, y_train, y_test
            # apply min max limits to the forecast
            _y_pred_values = min_vfunc(_y_pred_values,minmax[n][0]) if minmax[n][0] != None else _y_pred_values
            _y_pred_values = max_vfunc(_y_pred_values,minmax[n][1]) if minmax[n][1] != None else _y_pred_values
    
        #print("sliding_forecast: _y_pred_values =",_y_pred_values)
        
        index = dfXY.index if fit == True else index_pred
        _df_pred = pd.DataFrame(index=index) # keep the index from dfXYfc
        
        if fit == True:  
            # first observation
            i1_obs = 0 
            
            ytrain_col = _y +"_train"
            ytest_col  = _y +"_test"
            error_col = _y + "_pred_error"
            ypred_lower = ypred_col +"_lower"
            ypred_upper = ypred_col +"_upper"
              
            if Ntest > 0 : # fit true Ntest > 0 ... # metrics, errors, confidence 
                          
                errors=list(np.array(_y_pred_values) - np.array(_y_test_values))
                idx = dfXY.iloc[y_pred_idx].index # prediction indexes
                
                if debug == True:
                    print("slidng_forecast: _y =", _y)
                    print("slidng_forecast: ypred_col =", ypred_col)
                    print("sliding_forecast: _ytest_values = ",_y_test_values)
                    print("sliding_forecast: _pred_values = ",_y_pred_values)
                    print("sliding_forecast: errors = ", errors)
                    print("sliding_forecast: idx = ", idx)
                
                
                _df_pred[ytrain_col] = dfXY.iloc[i1_obs:i_initial_pred][_y] # y train
                _df_pred[ytest_col] = dfXY.iloc[i_initial_pred:][_y] # y test
                _df_pred[ypred_col] = np.nan
                _df_pred.loc[idx, ypred_col] = _y_pred_values # prediction value
                
                _df_pred.loc[idx,error_col] = errors
                
                # metrics dictionary ... statistics
                #N = Ntest if Ntest > 0 else 1
                
                #N = Ntest if Ntest > 0 else Ntest
                y_test_array = dfXY[_y].iloc[i_initial_pred:i_initial_pred+Ntest]
                y_pred_array = _df_pred.iloc[i_initial_pred:i_initial_pred+Ntest][ypred_col]
                         
                rmse_test = np.sqrt(MSE(y_test_array, y_pred_array))
                mae_test= MAE(y_test_array, y_pred_array )
                metrics[ypred_col] = {"RMSE":rmse_test, "MAE":mae_test}

                # confidence intervals
                #  returns error_lower, error_upper
                _df_pred, nh_err_lowerupper = forecast_confidence(_df_pred, alpha=alpha, error=error_col, Nhorizon = Nhorizon, method=ci_method, verbose=verbose, debug=debug)
                _df_pred[ypred_lower] = _df_pred[ypred_col] + _df_pred["error_lower"]
                _df_pred[ypred_upper] = _df_pred[ypred_col] + _df_pred["error_upper"]
                
                if debug == True:
                    print("nh_err_lowerupper = ",nh_err_lowerupper)
                    print(type(nh_err_lowerupper))
                ci[_y] = nh_err_lowerupper

                # Apply minmax to CI
                _df_pred[ypred_lower] = _df_pred[ypred_lower].apply(lambda x: min_func(x,minmax[0][0])) if minmax[0][0]!=None else  _df_pred[ypred_lower]
                _df_pred[ypred_upper] = _df_pred[ypred_upper].apply(lambda x: max_func(x,minmax[0][1])) if minmax[0][1]!=None else  _df_pred[ypred_upper]

                # drop error_lower and error_upper
                _df_pred=_df_pred.drop(["error_lower","error_upper"],axis=1)  
                
            else:    # fit==True, Ntest == 0   
                i_last_row= dfXY.index.size -1
                idx = dfXY.iloc[i_last_row:i_last_row + 1].index # set idx to last test idx
                errors = np.NaN  
                _df_pred[ytrain_col] = dfXY[_y] # y train
                _df_pred[ytest_col] = np.NaN # y test
                _df_pred[ypred_col] = np.NaN
                _df_pred.loc[idx,error_col] = errors
                
                # metrics
                rmse_test = np.NaN
                mae_test= np.NaN
                metrics[ypred_col] = {"RMSE":rmse_test, "MAE":mae_test}
                
                # confidence intervals
                _df_pred[ypred_lower] = np.NaN
                _df_pred[ypred_upper] = np.NaN
                
        # predict=True
        # package up predictions with indexes       
        if predict == True:
            idx = index_pred # set idx 
            if debug == True: 
                print("sliding_forecast: predict, index_pred =",index_pred)
            _df_pred[ypred_col] = np.nan
            _df_pred = pd.DataFrame(_y_pred_values, index = idx, columns=[ypred_col])
            
        # join predictions from each target _y 
        #  to wide df_pred frame
        # fit or predict == True    
        df_pred = df_pred.join(_df_pred) if n!=0 else _df_pred
        
    if fit == True:
        return dfXY_train, df_pred, metrics, m, history_i, history_t, m_fit, scaler, make_derived_attributes, make_lags, ci
    
    if predict == True:
        return df_pred
        
        ### end sliding_forecast
        
class sforecast:
    """Sforecast supports sliding (expanding) window fit and predict operations. 
    The fit operation is an out-of-sample train test fit, where the Ntest (swin_paramters:Ntest) defines the size of the training obervations
    and test observations. 
    
    The out-of-sample train-test methodology works as follows. The first training and test window (before sliding) has a training set comprised of total observations - Ntest and test set size
    Ntest. The model is retrained every Nhorizon steps, where Nhorizon defaults to 1. 
    After the first train/test operation, the training set window slides forward by Nhorizon observations. The training set increases by Nhorizon observations, and the 
    test set decreases by Nhorizon observations.

    The fitted transformers make_lags, scaler, and make_derived_attributes, transform the data during test or predict operations.
    The transformers
    generate lagged variables, derived variables, and together with exogenous variables are scaled. 
    These are subsuquently input to the fitted model to make a prediction.
    
    The last fit corresponds to all observation in the dfXY input DataFrame. 
    The fitted model, make_lags transform make_derived_attributes and scaler transform are returned in addition to test predictions, and index for each prediciton matching the input dataframe.

    Sforecast supports recursive (Nhorizon x 1-step) forecasts and (under development) direct forecasts (Nhorizon models) over the Nhorizon interval. 
    The predict operation makes N-step (N x 1-step) recursive forcasts, correspondng to the recursive/single_step fit operation or Nhorizon direct forecasts (under development) correspondng to a direct forecast fit operation.
    
    **__init__(self,  y="y", model=None, ts_parameters=None)**
     Recieves inputs defining the sliding forecast model including ML model, time-series sliding/expanding window hyper-parameters, and ML feature scaling.
        
        **Parameters**
            * y (String or List): single target (dependent) variable (str) or list of target variables.  
            * model (ML Model): The ML model to be used in the forecast operation. It can be a SK Learn model (model_type = "sk") or TensorFlow model (model_type = "tf"). This variable is ignored if model_type = "cm" (classical forecast moodel).  
            * model_type (String): indicates the type of model, "sk" (Sklearn), "cm" (statsmodel or pdarima), or "tf" (TensorFlow).  
            * swin_parameters (Dictionary): sliding window forecast model parameters.   
                * Nlags (Integer): Number of lags for all target variables and covariates. Lagged variables enable accounting for the auto-regressive properties of the timeseries. Defaults to 1.  
                * covars (List): List of covariate variables. If not already present, the y forecast variable(s) will be added to the covars list. Non-lagged covariates (lag = 0) are removed from the training variables to avoid leakage. Lag values > 0 and <= Nlags are included in the training variables.  
                * catvars (List): List of categorical variables. This input is only relevant for TensorFlow models and is ignored otherwise. Default = None.  
                * exenvars (List): List of exogenous (e.g., input variables) and enodogenous (e.g., derived variables). Exenvars is for continuous, variables. Categorical variables are contained in catvars, not in exenvars. Exenvars can be a list of lists, in which case it represents groups of variables that can be processed differently, for example by the TensorFlow model. For example, a TensorFlow models can process exenvars with a dense feed forward network and covariate lagged variables in an LSTM.  
                * Ntest (Integer) - number of predictions to make. Defaults to 1. Ntest > 0 will cause the sliding forecast to divide the observations into trainng and test. The first training will use the last Ntest observatios as the test set and the previous observations as the training set. Ntest can be set to 0 in which case all observations are used as the training set and there are no test statistics to maintain.  
                * alpha (Float): A number between 0 and 1 designating the donfidence interval spread. Defaults to 0.2 (80%).
                * Nhorizon (Integer) - n-step (i.e., Nhorizon) forecast. For example, the sliding/expanding window will move forward by Nhorizons after Nhorizon predictions. Default to 1.  
                * minmax (Tuple): Defaults to (None, None). Imposes and lower and upper limit on the forecast/predictions (and confidence intervals), respectively.
                * ci_method (String):
                    * The method used to estimate the confidence interval. Defaults to "linear" from the numpy percentile function. Choices are
                        * "inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased"    
                        * "minmax" - the min and max values observed errors  
                        * "tdistribution" - compute the t-distribution confidence interval   
                * horizon_predict_method (String): "single_step" or "direct". "single_step" indications predictions over the horizon window are recursive, meaning a single ML model with Nhorizon 1-step recursive predictions. "direct" indeicates that predictions over the horizon interval use the direct forecasting (Nhorizon models).
                * derived_attributes_transform: a transform for derived (endogenous/dependent) attributes. See example.ipynb for how to create derived endogenous attributes.
            * tf_params (Dictionary, optional) - TensorFlow parameters. Defaults to None.
                * Nepochs_i: Number of training epochs for the first (intitial training). Defaults to 10.  
                * Nepochs_t: Number of training (tuning) epochs for subsequent trainng, after the initial traiing. Defaults to 5.  
                * batch_size: Defaults to 32.  
                * lstm: defaults to False.  
            * cm_params (Dictionary): parameters for the classical forecast models, when model_type = "cm"
                * model (str): The supported models are "arima", "sarimax" and "auto_arima". Default is None.  
                * order (tuple): SARIMA and ARIMA order tuple contains the (p,d,q) parameters of the ARIMA and SARIMA models. Defaults to None.  
                * seasonal_order (tuple): sarima seasonal order (P,D,Q,m). Defaults to (0, 0, 0, 0).  
                * enforce_stationarity (Boolean): sarima. Defualuts to True.  
                * enforce_invertibility (Boolean): SARIMAX. Defaults to True.
                * start_p (int):  autoarima, starting p, lags. Defaults to 1.  
                * start_q (int): autoarima, starting q, ma error lags.  
                * d (Boolean): autoarima differencing paramter. Defaults to None, discovered.  
                * seasonal (Boolean): autoarima. Defaults to True.  
                * "max_p": autoarima, Default to None.  
                * "max_q": autoarima, Defaults to None.
                * test (string): autoarima stationarity test, Default ot "adf", automated Dickey-Fuller test.  
                * start_P (int): autoarima seasonal order. Defautls to 1. 
                * start_Q (int): autoarima seasonal order. Defautls to 1. seasonal ma (error) order.  
                * max_P: autoarima, Defaults to None.   
                * max_Q: autoarima. Defaults to None.  
                * m (int): autoarima seasonal period (number of observations (rows) corresponding to the season) Defaults to 12 auto arima,   
                * D (int): autoarima, sasonal difference. Defaults to None, discovered with seasonality = True.  
                * trace (Boolean): autoarima, defalults to True. Print model AIC.   
                * error_action (str): autoarima. Defaults to "ignore", don't want to know if an order does not work.   
                * suppress_warnings (Boolean): autoarima. Defaults to True.   
                * stepwise: autoarima. Defaults True, stepwise search.  
            * scale_parameters (Dictionary): input variables are scaled as designated by the parameters in the dictionary.  
                * mms_cols (str or list): Defaults to "all" in which case all input variables are scalled with the SKlearn MinMax scaler. If mms_cols = None, then no variables are scaled with the MinMax scaler. If mms_cols = list of columns, then the corresponding columns are scaled with the MinMax scaler.  
                * ss_cols (str or list): The ss_cols option takes precedence over mms_cols. Defaults to None. If ss_cols =  "all" all input variables are scalled with the SKlearn StandardScaler scaler. If xscale_params["ss_cols"] = list of columns, then the corresponding columns are scaled with StandardScaler. 
                * minmax (2-tuple): forecassts predictions and ci (confidence intervals) are constrained to fall between minmax[0] and minmax[1]. Defaults to (None, None), meaning no lower or upper limits are imposed on the predictions and confidence intervals.
        
    **State Variables**
        * metrics (dictionary of dictionaries):  {{y target variable): {MSE: number} , {MAE: number} } , ... }  
        * model (ML Model):  ML Model. After fitting the variable corresponds to the trained model trained on all observations.  
        * dfXYfit_last (DataFrame): the last row of the fitted dataframe. The dataframe includes covariate columns (note, unlagged covariates are not used for the fit or predict operations) and all columns (variables) used in the predict operation includind lagged covariates, exogenous variables, derived/endogenous variables, and categorical variables. Continuous variables (not covariates) are scaled according the scale_parameters input dictionary.  
        * df_pred (DataFrame): Dataframe containing the prediciton results. See sforecast.fit() for additional information.  

    """
    def __init__(self,  y:list, model:None, model_type:None, xscale_parameters = None, 
                 swin_parameters=None, tf_parameters=None, cm_parameters=None, scale_parameters=None, verbose = False, debug = False) -> None:
  
        # initialize sliding window parameters
        self.swin_params = {
            "covars": None,
            "catvars":None,
            "exogvars": None, # exogvars not includng derived variables
            "derived_attributes": None,
            "exenvars":None, # list of exogenous + engogenous (derived variables)
            "Ntest":1,
            "Nlags":1,
            "Nhorizon":1,
            "idx_start":None,
            "alpha":0.2,
            "ci_method":"linear",
            "minmax":(None,None),
            "horizon_predict_method":"single_step", # single-step (recursive) , direct
            "derived_attributes_transform":None,
            "derived_attributes":None
        }
        
        self.scale_params = {
            "mms_cols":"all",
            "ss_cols":None,
            "scaler":None,
        }
        
        self.tf_params = {
            "Nepochs_i":10,
            "Nepochs_t":5,
            "batch_size":32,
            "lstm":False
        }
        
        self.cm_params = {
            "model":"arima", # arima ,  saramiax, auto_arima
            "order":(1,1,1), # arima and saramiax
            "seasonal_order":(0, 0, 0, 0), # sarimax
            "enforce_stationarity":True, # sarimax
            "enforce_invertibility":True, # sarimax
            #"smoothing_level":0.6, # sarimax
            #"smoothing_trend":0.2, # sarimax
            #"initialization_method":"estimated", #sarimax
            "start_p":1, # auto arima
            "start_q":1, # auto arima
            "d":None, # auto arima
            "seasonal":True, # auto arima
            "max_p":None, # auto arima
            "max_q":None, # auto arima
            "test":"adf", # auto arima automated Dickey-Fuller test
            "start_P":1, # auto arima
            "start_Q":1, # auto arima
            "max_P":None, # auto arima
            "max_Q":None, # auto arima, 
            "m": 12, # auto arima, 
            "D":None, # auto arima, seasonal difference, discovered with seasonality = True
            "trace":True, # auto arima,  print model AIC 
            "error_action": "ignore", # auto arima, don't want to know if an order does not work
            "suppress_warnings":True, # auto_arima, stepwise search
            "stepwise": True # auto_arima, stepwise search
        }
        
        self.predict_params = {
            "index":None,
            "dfexogs": None,
            "Nperiods": 1,
            "make_lags": None,
            "make_derived_attributes":None
            }
        
        # sliding winddow parameters
        if swin_parameters != None:
            for k in swin_parameters:
                assert self.swin_params.__contains__(k), f'swin_parameters, {k}, is not valid'
                self.swin_params[k] = swin_parameters[k]
                
        if self.swin_params["Nhorizon"] == 1:
            assert self.swin_params["horizon_predict_method"] == "single_step" , f'horizon_predict_method = {self.swin_params["horizon_predict_method"]} not valid.Only horizon_predict_method = "single_step" is allowed when Nhorizon = 1'
        
        # y and covars
        # ensure y is iterable/list
        # ensure y is in covars and covars is iterable
        y = [y] if isinstance(y,str) else y
        if self.swin_params["covars"] == None: 
            self.swin_params["covars"]  = []
        self.swin_params["covars"] = [self.swin_params["covars"]] if isinstance(self.swin_params["covars"],str) else self.swin_params["covars"]
        for _y in  y: 
            if _y not in self.swin_params["covars"]: self.swin_params["covars"].append(_y)
            
        Nhorizon = self.swin_params["Nhorizon"]
        if self.swin_params["horizon_predict_method"] == "multi_step":
            assert Nhorizon > 1, f'Norizon = {Nhorizon} not valid. Norizon must be > 1 when horizon_predice_method = multi_step'
            
        ### Min Max ####   
        # ensure minmax is iterable ... 
        # same length as y ... 
        # if there is only one minmax pair then duplicate
        minmax=self.swin_params["minmax"]
        if isinstance(minmax,tuple): minmax = [minmax]
        _minmax = minmax.copy()
        _minmax = len(y)*_minmax if (len(_minmax) != len(y)) and (len(_minmax)==1) else _minmax
        assert len(_minmax) == len(y), f'length of minmax tuples = {len(minmax)}, must equal the number of y targets'
        self.swin_params["minmax"] = _minmax # replace the original minmax with the iterable minmax
            
        self.y = y                      # list of target variables or string name of target variable
        
        # model 
        #  for cm or sk ... each covar has a model ... tf one model generates all covars
        #     Nhorizon > 1 each time period within a horizon has a model
        #     at minimum covars contains y ... length of 1
        covars = self.swin_params["covars"]
        if ( model_type == "cm" or model_type == "sk") and (len(covars) > 1 or len(y) > 1): # cm len(y) can be greater than 1 w/o covars
            self.model = {} # default model is a dictionary
            for cv in covars: 
                self.model[cv] = model # clone the model in test_train_predict so copy by reference here
        else: # only one y or tf
            self.model = model

        # confidence interval dictionary 
        # confidence range for each y target, and time horizon from 1 to Nhorizion, {y1:[], y2:[], ...}
        # these confidences are estimated/measured during the fit process
        self.ci = {}
        
        self.model_type = model_type    # type of forecast model ... cm, sk, tf
        self.history_i=None             # TensorFlow training history
        self.history_t=None             # TensorFlow tunning for last tunning/training cycle
        self.idx_last_obs = None        # last observation index
        self.dfXYfit_last = None        # last observation for forcast ... scaled exogs, endogenous, and lags ... cats are not scaled
        self.metrics = None             # rmse and mae for each y prediction
        self.ci  = None                 # ci upper lower

        # Nhorizon        
        assert self.swin_params["Nhorizon"] >= 1, f'Nhorizon must be >= 1'
          
        # ensure exenvars is iterable 
        if self.swin_params["exenvars"] != None:
            self.swin_params["exenvars"] = [self.swin_params["exenvars"]] if isinstance(self.swin_params["exenvars"],str) else self.swin_params["exenvars"]
          
        # ensure exogvars is iterable 
        if self.swin_params["exogvars"] != None:
            self.swin_params["exogvars"] = [self.swin_params["exogvars"]] if isinstance(self.swin_params["exogvars"],str) else self.swin_params["exogvars"]
        
        # derived attribute names
        derived_attribute_names =  self.swin_params["derived_attributes_transform"]().get_derived_attribute_names() if self.swin_params["derived_attributes_transform"] != None else None
        self.swin_params["derived_attributes"] = derived_attribute_names
        
        if verbose or debug == True:
            print('sforecast.init: self.swin_params["derived_attributes] =', derived_attribute_names)
        
        # Exenvars
        # create exenvars if it doesnt already exist
        if self.swin_params["exenvars"] == None: # exenvars is None ... create it     
            if self.swin_params["exogvars"] == None:
                self.swin_params["exenvars"] = derived_attribute_names 
                
            else: # exogvars != None ... exenvars = [exogvars] + [derived_attribues]
                self.swin_params["exenvars"] = self.swin_params["exogvars"] if derived_attribute_names == None else self.swin_params["exogvars"] + derived_attribute_names
                
        # ensure exogenous and endogenous variables are in exenvars
        # ... above ensure exenvars exists if it is needed
        exenvars = self.swin_params["exenvars"]
        if self.swin_params["exenvars"] != None:
             #  first flatten exenvars
            exenvars_flat = [ex for exen in exenvars for ex in exen] if not isinstance(exenvars[0],str)  else exenvars  
            
             # ensure exogenous variables are in exenvars
            if self.swin_params["exogvars"] != None:
                for exog in self.swin_params["exogvars"]:
                    assert exog in exenvars_flat, f'exogvars variable {exog} is not in exenvars = {self.swin_params["exenvars"]}. exenvars must contain all exogenous variables ("exogvars") and endogenous variables "(derived variables)"'
            
            # ensure endogenous/derived variables are in exenvars
            if derived_attribute_names != None:
                for endog in derived_attribute_names:
                    assert endog in exenvars_flat, f'derived attributes variable {endog} is not in exenvars = {self.swin_params["exenvars"]}. exenvars must contain all endogenous  variables ("exogvars") and and endogenous variables "(derived variables)"'
           
        # exenvars
        if verbose or debug == True:
            print('sforecast.init: self.swin_params["exenvars"] = ', self.swin_params["exenvars"])
        
        assert self.swin_params["Ntest"] >= 0, f'Ntest = {self.swin["Ntest"]}, must be >= 0'
                
        # tensorflow parameters   ... nepochs_i, nepochs_t,      
        if tf_parameters != None:
            for k in tf_parameters:
                assert self.tf_params.__contains__(k), f'tf_parameters, {k}, is not valid'
                self.tf_params[k] = tf_parameters[k]
        
        # classic model paramters        
        if cm_parameters != None:
            for k in cm_parameters:
                assert self.cm_params.__contains__(k), f'cm_parameters, {k}, is not valid'
                self.cm_params[k] = cm_parameters[k]
            assert self.cm_params["model"] == "arima" or self.cm_params["model"] == "sarimax" or self.cm_params["model"] == "auto_arima", f'model = {self.cm_params["model"]} not valid. Valid models are "arima", "sarimax", "auto_arima"'
        
        # scale variable parameters ... normalize, standardize
        if scale_parameters != None:
            for k in scale_parameters:
                assert self.scale_params.__contains__(k), f'x_scale parameter = {k} is not valid'
                self.scale_params[k] = scale_parameters[k]

        assert model_type != "None" , f'model_type = {model_type} is not defined, model_type must be defined'

        assert model_type == "cm" or model_type == "sk" or model_type == "tf" , f'model_type = "{model_type}" not supported'

        assert self.y != None, f'y forecast variable(s) not specified'
        
        #assert self.swin_params["Ntest"] % self.swin_params["Nhorizon"] == 0, f'Ntest = {self.swin_params["Ntest"]} is not evenly divisable by Nhorizon = {self.swin_params["Nhorizon"]}'
        
        if model_type == "cm" and (self.cm_params["model"]=="arima" or self.cm_params["model"]=="sarimax" ) :
            assert self.swin_params["Nhorizon"]==1, f'Nhorizon > 1 not supported for model_type = arima or sarimax'    
            
        if model_type == "cm" and (self.cm_params["model"]=="auto_arima" and self.swin_params["derived_attributes_transform"] != None ) :
            assert self.swin_params["Nhorizon"]==1, f'Nhorizon > 1 not supported for model_type = auto_arima and derived attributes (exogenous variables) != None'
                
    def fit(self, dfXY, verbose=False, debug=False):
        """The fit operation is an out-of-sample train test fit, where the Ntest (swin_paramters:Ntest) defines the size of the training obervations and test observations.
        The model is retrained every Nhorizon steps. Nhorizon defaults to 1.
        After the first train/test operation, the training set window slides forward and the training set increases by Nhorizon observations and the test set decreases by Nhorizon observatsions.
        The last fit corresponds to all observation in the dfXY input DataFrame. 
        
        Args:
            dfXY (DataFrame): input DataFrame containing the target variable, covariates (optional), exogenous continuous variables (optional), and categorical variables (optional).
            
            verbose (bool): True or False. Verbose == True causes the printing of helpful information. Defaults to False.

        Returns:
            DataFrame with the forecast output including the following columns for each covariate.  
                * y_train: y training value at the corresponding observation for the initial window (before sliding). Values outside the initial window will be NaN. After the initial training, y values will then be shown under the y_test column.  
                * y_test: y value (truth) corresponding to the prediction.  
                * y_pred: y predicted (i.e. forecast) 
                * y_lower:  lower confidence limit  
                * y_upper:  upper confidence limit  
        """
        
        # dfXY
        assert isinstance(dfXY, pd.DataFrame)
        assert dfXY.index.has_duplicates == False, f'dfXY has duplicate index entries. Duplcate indexes are not allowed'
        
        # y
        if isinstance(self.y,str):
            assert self.y in dfXY.columns, f' y = {self.y} column is not contained in the input DataFrame'
        else:
            for _y in self.y:
                assert _y in dfXY.columns, f' "{_y}" column is not contained in the input DataFrame'
               
        # ensure all columns are either y, exogvars, covars, or catvars
        cols = list(dfXY.drop(self.y, axis=1).columns)
        exogvars = self.swin_params["exogvars"]
        covars = self.swin_params["covars"]
        catvars = self.swin_params["catvars"]
        for c in cols:
            if exogvars != None:
                if catvars == None:
                    assert c in exogvars or c in covars, f'dfXY column "{c}" is not specified as y, exogenous variable, covariate variable, or categorical variable.'
                else:
                    assert c in exogvars or c in covars or c in catvars, f'dfXY column "{c}" is not specified as y, exogenous variable, covariate variable, or categorical variable.'
            else:
                if catvars == None:
                    assert c in covars, f'dfXY column "{c}" iis not specified as y, exogenous variable, covariate variable, or categorical variable.'
                else:
                    assert c in covars or c in catvars, f'dfXY column "{c}" is not specified as y, exogenous variable, covariate variable, or categorical variable. '
            
        dfXYfit, self.df_pred, self.metrics, self.model, self.history_i, self.history_t, self.model_fit, \
                    self.predict_params["scaler"], make_derived_attributes, make_lags, self.ci = \
                    sliding_forecast(dfXY, y = self.y, fit = True,
                        model=self.model, model_type=self.model_type, 
                        swin_params = self.swin_params, 
                        cm_params = self.cm_params,
                        tf_params=self.tf_params, 
                        scale_params = self.scale_params,
                        verbose = verbose, debug=debug)
         
        self.predict_params["make_derived_attributes"] = make_derived_attributes
        
        self.predict_params["make_lags"] = make_lags
            
        if (self.model_type == "cm") and (self.cm_params["model"] != "auto_arima"):
            self.model = self.model_fit 
   
        if debug == True:
            print("sforecast.fit: make_lags =",make_lags)
        
        self.idx_last_obs = dfXYfit.tail(1).index[0]
        
        last_i = dfXYfit.index.size - 1
        self.dfXYfit_last = dfXYfit.iloc[last_i:last_i +1].copy() # last observation ... scaled exogs, endogenous, and lags ... cats are not scaled
        
        return self.df_pred
    
    def predict(self, Nperiods=1, dfexogs=None, dfcats=None, ts_period = None, verbose=False, debug=False):
        """Nperiod forecast next N_periods based on the model trainded during the fit operation. 
        
        Args:
            Nperiods (int): Indicates how many periods forward to forecast. Defaults to 1. 
            dfexog (DataFrame, optional): Defaults to None. If the fit includes exogenous variables (independent continous variables), then a dfexog input dataframe is required. The number of rows of dfexog must be <= Npriods and the columns are the same as exogvars.
            dfcats (DataFrame, optional): Defaults to None. This input object applies only to TensorFlow models. It is required if the fit includes categorical variables. The categorical variables must be encodeded (e.g., SK label encoder) in the same way as for the fit operation.
            ts_period (pandas time offset, optional): Defaults to None. Not required if the timeseries index is an integer. It is required if the timeseries index is a timestamp. The input is ignroed (a warning is issued) if the timeeries index is an integer.
            verbose (bool, optional): Print helpful information to standard out when True. Defaults to False.

        Returns:
            Dataframe including the forecast for each target variable.
        """
        if not isinstance(dfexogs,pd.DataFrame):
            if dfexogs != None: assert isinstance(dfexogs, pd.DataFrame) , "dfexogs input must be a Pandas DataFrame"
        else:
            self.predict_params["dfexogs"]=dfexogs
  
        # index
        idx_last_obs = self.idx_last_obs # last observation index
        index = None
        Nhorizon = self.swin_params["Nhorizon"]
        
        if isinstance(idx_last_obs,np.int64) or isinstance(idx_last_obs,int): # if index is integer then just add 1 to each successive index
            if ts_period != None:
                warnings.warn('ts_period ignored when timeseries has integer index (i.e., non datetime)')
            index = [np.arange(idx_last_obs + 1 ,idx_last_obs + Nhorizon + 1 ,1)]           
        else:  # index not int, assume it is datetime
            index = [pd.Timestamp(idx_last_obs)+ i*ts_period for i in range(1,Nperiods+1) ] if ts_period != None else None
                
        # Unclear index
        assert index != None, f'Unable to interpret prediction index. Specify ts_period (pandas offset) when dfXY period is timestamp.'
        
        self.predict_params["index"]=index
        
        # Nperiods
        Nhorizon = self.swin_params["Nhorizon"]
        if self.swin_params["Nhorizon"] > 1 and self.swin_params["horizon_predict_method"]=="multi_step":
            assert Nperiods <= Nhorizon , f'Nperiods = {Nperiods}, When "horizon_predict_method == "multi_step", Nperiods must be <= Nhorizon = {self.swin_params["Nhorizon"]}'
        if self.model_type == "cm" and ( self.cm_params["model"] == "arima" or self.cm_params["model"] == "sarimax"):
            assert Nperiods == 1 , f'Nperiods = {Nperiods} not supported. Nperiods must equal 1 for arima and sarimax'
        if  self.model_type == "cm" and self.cm_params["model"] == "auto_arima" and not self.swin_params["derived_attributes_transform"] == None:
            assert Nperiods == 1 , f'Nperiods = {Nperiods} not supported. Nperiods must equal 1 for auto_arima with derived/edogenous variables'
        self.predict_params["Nperiods"]=Nperiods
        
        
        # dfexogs
        #   ... ensure enough exogs are input
        if isinstance(dfexogs, pd.DataFrame):
            assert self.swin_params["exogvars"] != None, f'Invalid input. dfexogs not allowed if swin_params exogvars == None'
            assert list(dfexogs.columns) == self.swin_params["exogvars"] , f'dfexogs columns {list(dfexogs.columns)} must match (names and order) swin_params exogvars = {self.swin_params["exogvars"]}'
            assert dfexogs.index.size >= Nperiods, f'dfexogs number of rows = {dfexogs.index.size} is invalid. dfexogs must contain at least Nperiod = {Nperiods} rows'
  
        # dfcats
        #  ... tf may use catvars for categorical embeddings
        #  ... ensure the corrent number of catvars entered

        if isinstance(dfcats, pd.DataFrame):
            assert self.model_type == "tf" , f'dfcats only requred for model_type = "tf"'
            assert self.swin_params["catvars"] != None, f'Invalid input. dfcats not allowed if swin_params catvars == None'
            assert list(dfcats.columns) == self.swin_params["catvars"] , f'dfcats columns {dfcats.columns} must match (names and order) to swin_params catvars = {self.swin_params["catvars"]}'
            assert dfcats.index.size >= Nperiods , f'dfcats number of rows = {dfcats.index.size} is invalid. dfcats must contain at least Nperiod = {Nperiods} rows'
            self.predict_params["dfcats"]=dfcats
        else:
            assert dfcats == None, f'dfcats = {dfcats} not valid. A dfcats dataframe is input only for a TensorFlow model with categorical embeddings.'
  
        if debug == True: 
            print("sf_forecast: predict: idx_last_obs =",idx_last_obs)
            print("sf_forecast.predict: index =",index)
        
        self.predict_params["sf_forecast.predict: index"] = index
        self.predict_params["sf_forecast.predict: Nhorizon"] = Nhorizon
        
        predict_params = copy.deepcopy(self.predict_params)  # need to do deepcopy or transforms, e.g., lags will produce different forecast for successive calls
        
        df_pred = sliding_forecast(self.dfXYfit_last, y = self.y,  predict=True, fit=False,
                        model=self.model, model_type=self.model_type, 
                        swin_params = self.swin_params, cm_params = self.cm_params,
                        tf_params=self.tf_params, scale_params = self.scale_params,
                        predict_params = predict_params, verbose = verbose, debug=debug)
                        
        return df_pred
        