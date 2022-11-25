#from keras.optimizers import Optimizer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import t
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

#from keras.models import Model, clone_model

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm


def ci_percentile(errors, alpha, method="linear"):
    """_summary_

    Args:
        errors (Numpy Array): Array of errors
        alpha (Numeric): number from 0 to 1 indicating the confidence interval spread, e.g., 0.05 (95%)
        method (str, optional): _description_. Defaults to "linear", method = one of
            * ("inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased")
    Returns:
        tuple: error_lower, error_uppe
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
        tuple: error_lower, error_upper
    """
    
    n = len(errors)
    s = np.std(errors, ddof = 1)  # sample std dev, divisor= N-ddof, delta degrees of freedom ... = N-1 for sample std dev
    t_critical = t.ppf(1-alpha/2, df = n-1) # account for both tails, prob of each tail is alpha/2
    e_mean = errors.mean()
    SE = t_critical * s / np.sqrt(n)
    
    error_lower = e_mean - SE
    error_upper = e_mean + SE

    
    return error_lower, error_upper

def forecast_confidence(df,alpha,Nhorizon = 1, error="error", method="linear", verbose=False):
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
            * mumpy.percentil() function, method = one of
                * ("inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased")
            * "minmax" - the min and max values observed errors
            * "tdistribution" - compute the t-distribution confidence interval
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Dataframe: the input dataframe with error_lower and error_upper
    """
    
    df_error = pd.DataFrame(df[error])
    df_error = df_error.dropna() # only keep the horizon forecast periods, all others will be NA
    df_error["horizon_id"] = 1
    df_error["horizon_id"] = df_error["horizon_id"].cumsum() - 1 # 0, 1, 2, 3 ...
    df_error["horizon_id"] = df_error["horizon_id"]//Nhorizon  # horizon_id = 0 0 0 , 1 1 1,   ... 
    df_error["horizon"] = 1
    df_error["horizon"]=df_error.groupby("horizon_id")["horizon"].cumsum() # horizon = 1 , 2, 3 , 1, 2, 3, 1 , 2, 3, ...
    
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
            print(f'confidence method = {method}')
      
        df_error.loc[nh_idx,"error_lower"] = error_lower
        df_error.loc[nh_idx,"error_upper"] = error_upper

        if verbose == True: 
            print(f'\nhorizon = {nh}')  
            print(f'nh_idx = {nh_idx}')
            print(f'confidicence interval, error_lower ={error_lower}')     
            print(f'confidicence interval, error_upper ={error_upper}')    
            
    df = df.join(df_error[["error_lower","error_upper"]])
        
    return df

def nlag_covars(df:object,co_vars:list, N_lags:int) -> object:
    """ add covariat lags (nlag columns) to DataFrame

    Args:
        df (object): DataFrame containing covariates
        co_vars (list): list of columns (covariates) that will be added to the input dataframe
        N_lags (int): add df([col]).shift(i) from i to Nlags for each col in co_vars

    Returns:
        DataFrame: DataFrame of XY variables with the addition of lagged variables.
    """
    
    # co_vars := auto regressive varibles
    
    dfXY = df.copy()
    
    # ensure that co_vars is iterable
    co_vars = [co_vars] if isinstance(co_vars,str) else co_vars

    for n in np.arange(1,N_lags+1):
        arv_shift_vars = []
        for arv in co_vars:
            arv_shift_var = arv+"_m"+str(n)
            arv_shift_vars.append(arv_shift_var)
        dfXY[arv_shift_vars] = dfXY[co_vars].shift(n)
        
    return dfXY

def min_func(x,minvalue):
    """return x if greater than minvalue, otherwise return minvalue. Vectorized into min_vfunc for use by numpy.

    Args:
        x (numeric): number to compare to minvalue
        minvalue (numeric): minimum allowed value

    Returns:
        numeric: returns x if greater than minvalue, otherwise returns minvalue
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
        numeric: returns x if greater than minvalue, otherwise returns minvalue
    """
    if ~np.isnan(x):
        x = x if x < maxvalue else maxvalue
    return x

min_vfunc = np.vectorize(min_func)
max_vfunc = np.vectorize(max_func)

def predict_horizons(dfXYfc:object, y:list,  model:object,  covars:list,  i_initial_pred:int, 
            model_type="sk", catvars=None, exogvars=None, Nlags=None,
            Npred=1, Nhorizon=1, cm_params=None, tf_params=None, 
            Nobserve=None, verbose=False ) -> list:
    """Train and predict over horizons. The number of predictions = Npred. First prediction (forecast) corresponds to i-th (i_initial_pred) row of the input DataFrame.

    Args:
        dfXYfc (DataFrame): DataFrame with forecast variables.
        y (string, list): target variable string, or list of variables.
        model (object): reference/pointer to the forecast model
        covars (list): list of covariates
        i_initial_pred (int): i-th (integer) index of input DataFrame, corresponding to the first prediction. 
        model_type (str, optional): type of forecast model "cm" (classical), "sk" (SK Learn), or "tf" (TensorFlow). Defaults to "sk".
        catvars (list, optional): list of categorical variables. Defaults to None.
        exogvars (list, optional): list of exogenous variables. Defaults to None.
        Npred (int, optional): N predictions. Defaults to 1.
        Nhorizon (int, optional): Predict for Nhorizon before retraining or tuning (TensorFlow) the model. Defaults to 1.
        Nlags (int, optional): Number co-variate lags. Defaults to 1
        cm_params (Dictionary, optional): Classical model parameters. Defaults to None.
        
        tf_params (Dictionary, optional): TensorFlow parameters. Defaults to None.
            * Nepochs_i: Number of training epochs for the first (intitial training). Defaults to 10
            * Nepochs_t: Numbre of training (tuning) epochs for subsequent trainng, after the initial traiing. Defaults to 5.
            * batch_size: Defaults to 32
            * lstm: defaults to False
        Nobserve (int, optional): If Nobserve == None, then the first observation for training purposes is the first row of the input DataFrame. If Nobserve is an integer then Nobserve observations before the first prediction are selected. Defaults to None.
        verbose (bool, optional): Defaults to False.
        

    Returns:
        tuple: y_pred_values, y_test_values, y_pred_idx, m, model_fit (cm models), history_i (TensorFlow), history_t (TensorFlow)
            * y_pred_values (Numpy n-dimensional arrray): predicted values 
            * y_test_values (Numpy n-dimensional arrray): test_values (i.e., truth, actual observations)
            * m (model): ML forecast model. When model_type = "sk" or "tf" and "cm" AUTO_ARIMA , m corresponds to the fitted model, after the last traiing.
            * model_fit: Fitted model for the case of model_type == "cm" ARIMA and ARIMAX , otherwise = None.
            * history_i: TensorFlow training history (initial training)
            * history_t: TensorFlow training (tunning) history corresponding to the final (last) training.
    """
      
    X_cols = list(dfXYfc.columns) # remove y and covars next
    for cv in covars: X_cols.remove(cv)  # remove covariates from X ... if univariate then covars=_y
    Ncovars = len(covars)

    dfX = dfXYfc[X_cols]
    dfy = dfXYfc[y]
    dfXcats = dfX[catvars] if catvars != None else None 
    dfXlags = dfX.drop(catvars, axis=1) if catvars != None else dfX  # remove catvars from Xlags
    dfXlagsexogs = dfXlags
    dfXexogs = dfXlags[exogvars] if exogvars != None else None
    dfXlags = dfXlags.drop(exogvars, axis=1) if exogvars != None else dfXlags
    
    history_i = None # tensorflow initial history, first fit
    history_t = None # tensorflow tuning second, third, ... fit
    model_fit = None
    
    y_pred_idx = []
    
    # fit and predict horizon ... multiple horizons if Npred > Nhorizon
    # Nhorizon = 1 (nsteps) for classic models arima and saramax
    # Nhorizon can be greater than 1 (nstep prediction) for tf, sk, and auto_arima models
    for i in np.arange(i_initial_pred,i_initial_pred+Npred,Nhorizon):  
        
        # first observation
        i1_obs = 0 if Nobserve == None else i - Nobserve   # if Nobserve = None, then expanding window
        if verbose == True:
            print(f'training observations ilocs = {i1_obs}:{i-1}')
        
        Y_train = dfy.iloc[i1_obs:i].values # last train index =  i-1
        
        ###### initialize model and fit   
        if model_type == "cm" and cm_params["model"] == "arima":
            Y_train = dfy.iloc[i1_obs:i].values
            m = ARIMA(Y_train, order=cm_params["order"])
            pred_start = Y_train.shape[0]
            pred_end = pred_start
            model_fit = m.fit()
            
        if model_type == "cm"  and cm_params["model"] == "sarimax":
            Y_train = dfy.iloc[i1_obs:i].values
            if exogvars == None:
                m = SARIMAX(Y_train, order= cm_params["order"] , seasonal_order = cm_params["seasonal_order"],
                            enforce_stationarity = cm_params["enforce_stationarity"],
                            enforce_invertibility = cm_params["enforce_invertibility"] )
            else:
                Xexog_train = dfXexogs[i1_obs:i].values
                m = SARIMAX(Y_train, exog = Xexog_train, order= cm_params["order"] , seasonal_order = cm_params["seasonal_order"],
                            enforce_stationarity = cm_params["enforce_stationarity"],
                            enforce_invertibility = cm_params["enforce_invertibility"] )
            model_fit = m.fit(disp=False)
        
        if model_type == "cm" and cm_params["model"]== "auto_arima":
            # auto_arima initialize and fit model in one step
            if exogvars == None:
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
                
                
            else:     
                Xexog_train = dfXexogs[i1_obs:i].values
                m = pm.auto_arima(Y_train, exogenous = Xexog_train , 
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
                m.fit(Y_train,exogenous = Xexog_train) # fit model in place ... similar to sklearn ... different than arima and sarimax
             
        if model_type == "sk":
            m = clone(model)  # clone SKLearn untrained model
            X_train = dfX.iloc[i1_obs:i].values  # last train index is i-1   
            m.fit(X_train,Y_train)
            
        if model_type == "tf":
            Xlags_train = dfXlags[i1_obs:i].values
            # X cats is a list of dataframes ... embeddins will be a list of inputs
            Xcats_train_list =  [dfXcats[c][i1_obs:i].values for c in catvars] if catvars != None else None
            Xlagsexogs_train = dfXlagsexogs[i1_obs:i].values if exogvars != None else None
            Xexogs_train =  dfXexogs[i1_obs:i].values if exogvars != None else None
            Xlags_train = dfXlags[i1_obs:i].values

            if  tf_params["lstm"]==False:
                if exogvars != None and catvars != None:
                    X_train = [Xlagsexogs_train] + Xcats_train_list
                if exogvars == None and catvars != None:
                    X_train = [Xlags_train] + Xcats_train_list
                else:
                    X_train = Xlags_train
            elif tf_params["lstm"] == True:
                if catvars!= None and exogvars!= None:
                    X_train = [Xlags_train.reshape(Xlags_train.shape[0], Nlags, Ncovars)] + [Xexogs_train] + Xcats_train_list
                elif catvars==None and exogvars!=None:
                    X_train = [Xlags_train.reshape(Xlags_train.shape[0], Nlags, Ncovars)] + [Xexogs_train]
                elif catvars!=None and exogvars==None:
                    X_train = [Xlags_train.reshape(Xlags_train.shape[0], Nlags, Ncovars)] + Xcats_train_list
                else:
                    X_train = Xlags_train
             
            if i == i_initial_pred:
                m=model # set a pointer to the model ... note that cloning will require compiling and implementng redundent TF logic ... dont do that here
                if verbose == True: print("Nepochs_i =",tf_params["Nepochs_i"])
                history_i=m.fit(X_train,Y_train,epochs=tf_params["Nepochs_i"], batch_size = tf_params["batch_size"], verbose=0)
            else:
                # this will tune the previously trained model ... Nepochs_t < Nepochs_i
                if verbose == True:  print("Nepochs_t =",tf_params["Nepochs_t"])
                history_t=m.fit(X_train,Y_train,epochs=tf_params["Nepochs_t"], batch_size =tf_params["batch_size"], verbose=0)      
        
        #### X_test and Xexogs_test   
        if model_type == "sk":
            X_test = dfX.iloc[i:i + Nhorizon].values 
         
        elif model_type == "tf":
            Xlags_test = dfXlags.iloc[i:i+Nhorizon].values
            Xcats_test_list =  [dfXcats[c].iloc[i:i+Nhorizon].values for c in catvars] if catvars != None else None
            Xlagsexogs_test = dfXlagsexogs.iloc[i:i+Nhorizon].values if exogvars != None else None
            Xexogs_test =  dfXexogs[i:i+Nhorizon].values if exogvars != None else None
            Xlags_test = dfXlags[i:i+Nhorizon].values

            if tf_params["lstm"]==False:
                if exogvars != None and catvars != None:
                    X_test = [Xlagsexogs_test] + Xcats_test_list
                if exogvars == None and catvars != None:
                    X_test = [Xlags_test] + Xcats_test_list
                else:
                    X_test = Xlags_test
            elif tf_params["lstm"] == True:
                if catvars!=None and exogvars!=None:
                    X_test = [Xlags_test.reshape(Xlags_test.shape[0], Nlags, Ncovars)] + [Xexogs_test] + Xcats_test_list
                elif catvars==None and exogvars!=None:
                    X_test = [Xlags_test.reshape(Xlags_test.shape[0], Nlags, Ncovars)] + [Xexogs_test]
                elif catvars!=None and exogvars==None:
                    X_test = [Xlags_test.reshape(Xlags_test.shape[0], Nlags, Ncovars)] + Xcats_test_list
                else:
                    X_test = Xlags_test.reshape(Xlags_test.shape[0], Nlags, Ncovars)
                    
        elif model_type == "cm":
            X_test = dfX.iloc[i:i + Nhorizon].values 
            if exogvars != None:
                 Xexogs_test = dfXexogs[i:i+Nhorizon].values
                 
        #### predict horizons   
        
        if model_type == "cm":
            if  cm_params["model"] == "arima" or cm_params["model"] == "sarimax" :
            
                # predict by calling forecast from the fitted model
                if exogvars == None:
                    y_pred = model_fit.forecast()[0]
                else:
                    y_pred = model_fit(exog=Xexogs_test).forecast()[0]
                    
                y_pred = np.array([y_pred]).reshape(1,1)
            
            if cm_params["model"]== "auto_arima":
            
                # call forecast on the fitted model m
                n_periods = X_test.shape[0]
                print("n_periods = ",n_periods)
                if exogvars == None:
                    y_pred = m.predict(n_periods)
                    
                else:
                    y_pred = m.predict(n_periods, exogenous = Xexogs_test)
                 
                # reshape for consistenty with y_test, eg ... error = y_test - y_pred 
                y_pred = np.array([y_pred]).reshape(n_periods,1)    

            
        # sklearn and tensorflow
        else:
            y_pred = m.predict(X_test) # prediction ... Nhorizon = rows of X_test
            
        y_test = dfy.iloc[i:i + Nhorizon].values
        
        # put pred and test in nd arrays
        
        
        y_test_nda = np.append(y_test_nda,y_test,axis=0) if i != i_initial_pred else y_test
        y_pred_nda = np.append(y_pred_nda,y_pred,axis=0) if i != i_initial_pred else y_pred
            
        # save indexes to put back into original data frame
        # if Nhorizon not evenly divided into Npred then there will idx mismatch
        #    _n_idx takes care of potential idx mismatch for such a case
        _n_idx = y_test.shape[0]
        _pred_idx = list(np.arange(i,i + _n_idx,1))
        
        y_pred_idx+=_pred_idx # append new prediction indexes
        
        return_tuple = None 
        if model_type == "cm":
            return_tuple = y_pred_nda, y_test_nda, y_pred_idx, m ,model_fit
        if model_type == "sk":
            return_tuple = y_pred_nda, y_test_nda, y_pred_idx, m 
        elif model_type =="tf":
            return_tuple = y_pred_nda, y_test_nda, y_pred_idx, m , history_i, history_t
            
    return return_tuple  ### predict_horizons ###

def sliding_forecast(dfXY, y, model, model_type="sk", covars=None, catvars=None, exogvars= None, minmax=(None,None),
                     Nlags=1, Nobserve=None, Npred = 1,  Nhorizon=1, i_start = None, tf_params=None, cm_params=None,
                     idx_start=None, alpha = 0.2, ci_method="linear", mms_cols = None, ss_cols=None,  verbose = False):
    
    '''Receives as input DataFrame of X (exogenous + co-vvariates)  and Y (single or multiple forecasts), and an untrained model of type SKLearn or TensorFlow.

    Args:
    
        dfXY (DataFrame): DataFrame containing x and y (target) variables. The data frame is expected to be sorted in ascending order, and may or may not have a time index.
        
        model (ml model): An ML model either from SKLearn or TensorFlow. 
        
        model_type (str): Default = "sk" or "tf". Default = "sk".
        
        Nobserve (int): number of observations to include in the training data (counting back from the first prediction).  
        
        covars (list):  List of co-variates. If not already present, the y forecast variable(s) will be added to the co_vars list. Non-lagged co-variates (lag = 0) are removed from the training variables to avoid leakage. Lag values >0 and <= Nlags are included in the training variables. Defaults to None.
        
        catvars (list):  List of categorical variables, only relevant for TensorFlow models. Default = None.
        
        exogvars (list):  List of exogenous (continuous) variables. Note, in sf_forecast exogenous categorical variables are contained in cat_vars, not in exog_vars, though they are both a type of exogenous variable. Default = None. For the case of TensorFlow models, exog_vars are processed differently than covariate lagged variables. For example, exog_vars could be input to a dense network, while covariates could be input to an LSTM network, in which case require different formats. Use of exog_vars is optional. For example, if all continuous variables (e.g., exog_vars, and covariable lags) are input to a dense network, then there is no need to specifically identify the exog_vars.
    
        Nlags (int): For each of the covariates, add a lagged variable to the training from 1 to Nlags . Lagged variables account for the auto-regressive properties of the correspondng variable(s) in the ML training. Defaults to 1.
    
        Npred (int): number of predictions to make. Defaults to 1. 
        
        alpha (float): A number between 0 and 1 designating the donfidence interval spread. Defaults to 0.2 (80%).
    
        Nhorizon (int):  N-step (i.e., Nhorizon) forecast. For example, the sliding (or expanding) window will move forward by Nhorizons after Nhorizon predictions. Default to 1.
        
        tf_params: afjdsfjdl
        
        statsmodel: dsfsafl
        
        minmax (tuple): dsfsadlfjasdlfj
    
        i_start (int):  The first prediction, where i corresponds to the DataFrame iloc (i-th index). Defaults to None, in which case the first prediction = last observation - Npred + 1. Defaults to None.
    
        idx_start (int): The first prediction specified by the DataFrame index. idx_start takes presidence over i_start. Defaults to None.
    
        ci_method (string): The method used to estimate the confidence interval. Defaults to "linear" from the numpy percentile function. Choices are as follows.   
        
            * method from the mumpy.percentile function, one of  
            
                * "inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased"
            
                * "minmax" - the min and max values observed errors
                
                * "tdistribution" - compute the t-distribution confidence interval
                

        mms_cols (str or list): Defaults to "all" in which case all input variables are scalled with the SKlearn MinMax scaler. If xscale["mms_scale"] = None, then no variables are scaled with the MinMax scaler. If xscale["mms_cols"] = list of columns then the corresponding columns are scaled with the MinMax scaler.
                
        ss_cols (str or list): The ss_cols option takes precedence over mms_cols. Defaults to "all" in which case all input variables are scalled with the SKlearn StandardScaler scaler. If xscale["ss_scale"] = None, then no variables are scaled with the StandardScaler scaler. If xscale["ss_cols"] = a list of columns then the corresponding columns are scaled with StandardScaler.
            
        verbose: True or False. Defaults to False.

    Returns:
        tuple: dfXYfc, df_pred, metrics, m, history_i, history_t, model_fit 
            * dfXYfc: XY forecast DataFrame including lagged variables
            * df_pred: predictions. See documentation for sforcast.forecast.
            * metrics: Dictionary containg MSE and MAE for the corresponding predictions
            * m (model): ML forecast model. When model_type = "sk" or "tf" and "cm" AUTO_ARIMA , m corresponds to the fitted model, after the last traiing.
            * model_fit: Fitted model for the case of model_type == "cm" ARIMA and ARIMAX , otherwise = None.
            * history_i: TensorFlow training history (initial training)
            * history_t: TensorFlow training (tunning) history corresponding to the final (last) training.
   '''
    
    # if y not in covars then add to covars
    # ensure y is iterable
    # ensure _co_vars is iterable
    y = [y] if isinstance(y,str) else y
    if covars == None: covars = []
    covars = [covars] if isinstance(covars,str) else covars
    for _y in  y: 
        if _y not in covars: covars.append(_y)   
        
    # ensure minmax is iterable ... same length as y ... if there is only one minmax pair then duplicate
    if isinstance(minmax,tuple): minmax = [minmax]
    _minmax = minmax.copy()
    _minmax = len(y)*_minmax if (len(_minmax) != len(y)) and (len(_minmax)==1) else _minmax

    # assertions
    assert len(y)==len(_minmax), f'length of minmax tuples = {len(minmax)}, should equal the number of y targets'
    
    # copy the input dataframe to ensure the original is not modified
    dfXYfc = dfXY.copy()
    
    # covariate lags
    # statsmodels manage lags
    if model_type != "cm":
        if Nlags > 0:
            dfXYfc = nlag_covars(dfXY,covars, Nlags)
            dfXYfc = dfXYfc.iloc[Nlags:].copy()  # delete the first Nlags rows since they will contain NaNs

    # Scale the X inputs 
    #  covar lags (not unlagged) and exogenous variables are scaled ..  
    #  y (target variables) not scaled
    #  do not scale if model_type = "cm"
    if model_type != "cm":
        dfXYfc_scaled = dfXYfc # by detault dfXYfc_scaled = dfXYfc
        if mms_cols != None:
            if (mms_cols == "all") & (ss_cols != "all"): 
                mms_scaler =  MinMaxScaler()
                dfXYfc_scaled = mms_scaler.fit_transform(dfXYfc).copy()
                dfXYfc_scaled = pd.DataFrame(dfXYfc_scaled, columns = dfXYfc.columns, index = dfXYfc.index)
            elif mms_cols != "all":
                mms_scaler =  MinMaxScaler()
                dfXYfc_scaled = dfXYfc.copy
                dfXYfc_scaled[mms_cols] = mms_scaler.fit_transform(dfXYfc_scaled[mms_cols])
            dfXYfc_scaled[y] = dfXYfc[y]
        
        if ss_cols != None:
            if ss_cols == "all":
                ss_scaler = StandardScaler()
                dfXYfc_scaled = ss_scaler(dfXYfc).copy()
                dfXYfc_scaled = pd.DataFrame(dfXYfc_scaled, columns = dfXYfc.columns, index = dfXYfc.index)
            else:
                ss_scaler = StandardScaler()
                dfXYfc_scaled = dfXYfc.copy()
                dfXYfc_scaled[ss_cols] = ss_scaler.fit_transform(dfXYfc_scaled[ss_cols])
            dfXYfc_scaled[y] = dfXYfc[y]

        dfXYfc = dfXYfc_scaled # replaces the original dfXYfc DataFrame
    
    # sliding window variables
    Ntobs = dfXYfc.index.size # total observations
    
     ##### Prediction  ######
    # idx_start takes precedence over i_start
    # initial = initial prediction
    i_initial_pred = Ntobs - Npred if i_start == None else i_start
    i_initial_pred = (dfXYfc.loc[:idx_start].index.size)-1 if idx_start != None else i_initial_pred  # i_nitial starts at 0, subtract 1
    metrics={}
    df_pred=pd.DataFrame()
    
    # first observation
    i1_obs = 0 if Nobserve == None else i_initial_pred - Nobserve
    
    history_i=None #  initial model history (first fit) ... tensorflow
    history_t=None # tune model history (second, third, ... fit) ... tensorflow
    model_fit=None
    

    # TensorFlow            
    if model_type == "tf":
        #### Nhorizon predict forward Nhorizon
        # pred all targets (all _y in y) at once
        y_pred_values, y_test_values, y_pred_idx, m, history_i, history_t = predict_horizons(dfXYfc, y, model, 
                    covars=covars, i_initial_pred=i_initial_pred, model_type=model_type,  catvars=catvars,                          
                    exogvars=exogvars, Npred=Npred, Nhorizon=Nhorizon, Nlags=Nlags,tf_params=tf_params,
                    Nobserve=Nobserve, verbose=False)
    
    
    for n,_y in enumerate(y):
        # SK Learn Model
        # predict for every _y
        # sk models support only univariate prediction
        if model_type == "sk":
            # predict each indvidual _y independently
            _y_pred_values, _y_test_values, y_pred_idx, m = predict_horizons(dfXYfc, _y, model, 
            covars=covars, i_initial_pred=i_initial_pred, model_type=model_type,  catvars=catvars,                          
            exogvars=exogvars, Npred=Npred, Nhorizon=Nhorizon, Nlags=Nlags, tf_params=tf_params,
            Nobserve=Nobserve, verbose=False)
            
        # classic forecast models arima and sarimax (fyi - see above for auto arima)
        # support only 1-step (Nhorizon = 1), univariate predictition (for every _y)
        elif model_type == "cm":
            _y_pred_values, _y_test_values, y_pred_idx, m, model_fit = predict_horizons(dfXYfc, y, model, 
                    covars=covars, i_initial_pred=i_initial_pred, model_type=model_type,  catvars=catvars,                          
                    exogvars=exogvars, Npred=Npred, Nhorizon=Nhorizon, Nlags=Nlags,tf_params=tf_params,
                    Nobserve=Nobserve, cm_params=cm_params, verbose=False)
        
        # tf prediction done for all y's simultaneosly above (outside of loop) ... manage the variable inside loop    
        elif model_type == "tf":
            _y_pred_values = y_pred_values[:,n]
            _y_test_values= y_test_values[:,n]

        ypred_col = _y +"_pred"
        ytrain_col = _y +"_train"
        ytest_col  = _y +"_test"
        error_col = _y + "_pred_error"

        # y_pred, y_train, y_test
        # apply min max limits to the forecast
        _y_pred_values = min_vfunc(_y_pred_values,_minmax[n][0]) if _minmax[n][0] != None else _y_pred_values
        _y_pred_values = max_vfunc(_y_pred_values,_minmax[n][1]) if _minmax[n][1] != None else _y_pred_values
    
        errors=list(np.array(_y_test_values) - np.array(_y_pred_values))
        _df_pred = pd.DataFrame(index=dfXYfc.index) # keep the index from dfXYfc
        idx = dfXYfc.iloc[y_pred_idx].index # prediction indexes
        _df_pred[ytrain_col] = dfXYfc.iloc[i1_obs:i_initial_pred][_y] # y train
        _df_pred[ytest_col] = dfXYfc.iloc[i_initial_pred:][_y] # y test
        _df_pred[ypred_col] = np.nan
        _df_pred.loc[idx, ypred_col] = _y_pred_values # prediction values 
        
        _df_pred.loc[idx,error_col] = errors

        # metrics dictionary ... statistics
        y_test_array = dfXYfc[_y].iloc[i_initial_pred:i_initial_pred+Npred]
        y_pred_array = _df_pred.iloc[i_initial_pred:i_initial_pred+Npred][ypred_col]
        rmse_test = np.sqrt(MSE(y_test_array, y_pred_array))
        mae_test= MAE(y_test_array, y_pred_array )
        metrics[ypred_col] = {"RMSE":rmse_test, "MAE":mae_test}

        # confidence intervals
        _df_pred = forecast_confidence(_df_pred,alpha=alpha, error=error_col, Nhorizon=Nhorizon, method=ci_method, verbose=verbose)
        ypred_lower = ypred_col +"_lower"
        ypred_upper = ypred_col +"_upper"
        _df_pred[ypred_lower] = _df_pred[ypred_col] + _df_pred["error_lower"]
        _df_pred[ypred_upper] = _df_pred[ypred_col] + _df_pred["error_upper"]

        # Apply minmax to CI
        _df_pred[ypred_lower] = _df_pred[ypred_lower].apply(lambda x: min_func(x,_minmax[0][0])) if _minmax[0][0]!=None else  _df_pred[ypred_lower]
        _df_pred[ypred_upper] = _df_pred[ypred_upper].apply(lambda x: max_func(x,_minmax[0][1])) if _minmax[0][1]!=None else  _df_pred[ypred_upper]

        # drop error_lower and error_upper
        _df_pred=_df_pred.drop(["error_lower","error_upper"],axis=1)    
    
        # join to wide df_pred frame
        df_pred = df_pred.join(_df_pred) if n!=0 else _df_pred

    return dfXYfc, df_pred, metrics, m, history_i, history_t, model_fit    ### sliding_forecast ###


class sforecast:
    """Sliding window forecast model.
    
    **__init__(self,  y="y", model=None, ts_parameters=None)**
     Recieves inputs defining the sliding forecast model including ML model, time-series sliding/expanding window hyper-parameters, and ML feature scaling.
        
        Args:
            y (str or list): Forecast (dependent) variable(s)
            
            model (ML Model): SK Learn ML mode, tensorflow, or None (if statsmodel)
            
            model_type (str): indicates the type of model, "sk" (Sklearn), "cm" (statsmodel), or "tf" (TensorFlow).
            
            win_parameters (dictionary): Dictionary of sliding/expanding window forecast model hyperparameters. Defaults to "None".
            
                Nobserve (int): number of observations to include in the training data (counting back from the first prediction). Defaults to None (unspecified), expandng window.
            
                Nlags (int): Add a lagged variable to the training from 1 to Nlags . Lagged variables enable accounting for the auto-regressive properties of the correspondng variable(s) in the ML training. Defaults to 1.
            
                covars (list):  List of co-variate variables. If not already present, the y forecast variable(s) will be added to this co_vars list. Non-lagged co-variates (lag = 0) are removed from the training variables to avoid leakage. Lag values >0 and <= Nlags are included in the training variables. Defaults to None.
                
                catvars (list):  List of categorical variables, only relevant for TensorFlow models. Default = None.
                
                exogvars (list):  List of exogenous (and continuous) variables. Note, exogenous categorical variables are contained in cat_vars, not in exog_vars, though they are both a type of exogenous variable. Default = None. For the case of TensorFlow models, exog_vars are processed differently than covariate lagged variables. For example, exog_vars could be input to a dense network, while covariates could be input to an LSTM network. Use of exog_vars is optional. For example, if all continuous variables (e.g., exog_vars, and covariable lags) are input to a dense network, then there is no need to specifically identify the exog_vars.
            
                Npred (int): number of predictions to make. Defaults to 1. 
                
                alpha (float): A number between 0 and 1 designating the donfidence interval spread. Defaults to 0.2 (80%).
            
                Nhorizon (int):  n-step (i.e., Nhorizon) forecast. For example, the sliding/expanding window will move forward by Nhorizons after Nhorizon predictions. Default to 1. Defaults to 1
            
                i_start (int):  The first prediction where i corresponds the DataFrame iloc (i-th index). Defaults to None, in which case the first prediction = last observation - Npred + 1. Defaults to None.
                
                minmax (tuple): d;lsfasd;fjla;
            
                idx_start (int): The first prediction specified as the dataframe index. idx_start takes presidence over i_start. Defaults to None.
            
                ci_method (string): The method used to estimate the confidence interval. Defaults to "linear" from the numpy percentile function. Choices are as follows. 
                
                    * method from the mumpy percentile function, one of 
                    
                        * "inverted_cdf",  "averaged_inverted_cdf", "inverted_cdf", "averaged_inverted_cdf","closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased ", "normal_unbiased"  
                        
                    * "minmax" - the min and max values observed errors  
                    
                    * "tdistribution" - compute the t-distribution confidence interval  
                    
            tf_parameters (dictionary): adfjslfl
            
                fsdf;sj
                
                al;fjs;lfslfd
                
                a;lfjsdlfjl
                
            cm_params (dictionary): dsfdsfa
            
                model (str): The supported models are "arima", "sarimax" and "autoarima". THe Default is None.
                
                order (tuple): The order tuple contains the (p,d,q) parameters of the ARIMA and SARIMA models. The Default is None.
                
                seasonal_order (tuple): sarima seasonal order (P,D,Q,m). Defaults to (0, 0, 0, 0).
            
                enforce_stationarity (Boolean): sarima. Defualuts to True, 
            
                enforce_invertibility (Boolean):True, # sarimax
                
                start_p (int):  autoarima, starting p, lags. Defaults to 1.
                
                start_q (int): autoarima, starting q, ma error lags
                
                d (Boolean): autoarima differencing paramter. Defaults to None, discovered.
                
                seasonal (Boolean): autoarima. Defaults to True, 
                
                "max_p": autoarima, Default to None.
                
                "max_q": autoarima, Defaults to None, # auto arima
            
                test (string): autoarima stationarity test, Default ot "adf", automated Dickey-Fuller test
                
                start_P (int): 1, # autoarima, seasonal order
                
                start_Q (int): 1, autoarima, seasonal ma (error) order.
                
                max_P: autoarima, Defaults to None.
                
                max_Q: autoarima. Defaults to None.
                
                m (int): autoarima seasonal period (number of rows, i.e., obswervations) Defaults to 12 auto arima, 
            
                D (int): autoarim, sasonal difference. Defaults to None, discovered with seasonality = True
            
                trace (Boolean): autoarima True, # auto arima, print model AIC 
            
                error_action (str): autoarima. Defaults to "ignore", don't want to know if an order does not work
            
                suppress_warnings (Boolean): autoarima. Defaults to True.
            
                stepwise: autoarima. Defaults True, stepwise search
            
            xscale (dictionary): input variables are scaled as designated by the parameters in the dictionary.
            
                mms_cols (str or list): Defaults to "all" in which case all input variables are scalled with the SKlearn MinMax scaler. If xscale["mms_scale"] = None, then no variables are scaled with the MinMax scaler. If xscale["mms_cols"] = list of columns, then the corresponding columns are scaled with the MinMax scaler.
                
                ss_cols (str or list): The ss_cols option takes precedence over mms_cols. Defaults to "all" in which case all input variables are scalled with the SKlearn StandardScaler scaler. If xscale["ss_scale"] = None, then no variables are scaled with the StandardScaler scaler. If xscale["ss_cols"] = list of columns, then the corresponding columns are scaled with StandardScaler.
            
            minmax (2-tuple): forecassts predictions and ci (confidence intervals) are constrained to fall between minmax[0] and minmax[1], given the corresponding min or max is != None. Defaults to (None, None).
        
    
    **State Variables**
    
        metrics (dictionary of dictionaries):  {y (key): {MSE: number} , {MAE: number} , ... }
        
        model (ML Model):  ML Model. After forecasting (i.e., fitting) corresponds to the final state of the model at the last window position.
        
        ts_params (dictionary): dictionary of sliding/expanding window model hyper-parameters. See __init()__ documentation or further information.
        
        y (list): list of forecast variables
        
        dfXYfc  (DataFrame): the initial DataFrame with all initial rows, X, Y variables, and the addition of lagged variabiles minus initial rows with NaNs due to creating lagged variables (i.e., shifting).
        
        df_pred (DataFrame): Dataframe containing the prediciton results. See sforecast.forecast() for additional information.

    """
    def __init__(self,  y:list, model:None, model_type:None, xscale = None, tswin_parameters=None, tf_parameters=None, cm_parameters=None) -> None:
  
        # initialize sliding window parameters
        self.tswin_params = {
            "covars": None,
            "catvars":None,
            "exogvars":None,
            "Nobserve":None,
            "Npred":1,
            "Nlags":1,
            "Nhorizon":1,
            "i_start":None,
            "idx_start":None,
            "alpha":0.2,
            "ci_method":"linear",
            "minmax":(None,None)
        }
        
        self.xscale = {
            "mms_cols":"all",
            "ss_cols":None
        }
        
        self.tf_params = {
            "Nepochs_i":10,
            "Nepochs_t":5,
            "batch_size":32,
            "lstm":False
        }
        
        self.cm_params = {
            "model":"arima", # arima or saramiax
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
        
        self.y = y
        self.ml_model = model
        self.ml_model_type = model_type
        self.history_i=None
        self.history_t=None
        
        self.model = model
        
        if tswin_parameters != None:
            for k in tswin_parameters:
                assert self.tswin_params.__contains__(k), f'tswin_parameters, {k}, is not valid'
                self.tswin_params[k] = tswin_parameters[k]
                
        if tf_parameters != None:
            for k in tf_parameters:
                assert self.tf_params.__contains__(k), f'tf_parameters, {k}, is not valid'
                self.tf_params[k] = tf_parameters[k]
                
        if cm_parameters != None:
            for k in cm_parameters:
                assert self.cm_params.__contains__(k), f'cm_parameters, {k}, is not valid'
                self.cm_params[k] = cm_parameters[k]
        
        if xscale != None:
            for k in xscale:
                assert self.xscale.__contains__(k), f'xscale parameter = {k} is not valid'
                self.xscale[k] = xscale[k]

        assert model_type != "None" , f'model_type = {model_type} is not defined, model_type must be defined'

        assert model_type == "cm" or model_type == "sk" or model_type == "tf" , f'model_type = "{model_type}" not supported'

        assert self.y != None, f'y forecast variable(s) not specified'
        
        if model_type == "cm" and  self.cm_params["model"] == "arima":
            assert self.tswin_params["Nhorizon"]==1, f'Nhorizon is set to {self.tf_params.tswin_params["Nhorizon"]}, ARIMA model suports forecast horizon of 1, Nhorizon = 1'
        
        if model_type != "cm":
            assert self.ml_model != None, f'model cannot be {model}, ML model must be defined if model_type = "sk" or model_type = "tf" '
        
        assert self.ml_model_type == "sk" or self.ml_model_type == "tf" or self.ml_model_type =="cm", f'model type = {self.ml_model_type}, ml_model_type = "tf" or "sf" or "cm" is valid'
        
        if model_type == "cm" and (self.cm_params["model"]=="arima" or self.cm_params["model"]=="sarimax"):
            
            assert self.tswin_params["Nhorizon"]==1, f'if model_type = "cm" and model = "arima" or "sarimax",  Nhorizon must be = 1'
             
            if not isinstance(self.y,str):
                assert len(y) == 1, f'y = {y}, not valid. Only one target variable is supported with model_type = {model_type}'  
                

    def forecast(self, dfXY, verbose=False):
        """ML training (on training window) and predictions ("forecast") over the horizon (1 or more observation rows)

        Args:
            dfXY (DataFrame): input DataFrame containing X training variables (exogenous) and Y covariates.
            verbose (bool): True or False. Verbose == True causes the printing of helpful information. Defaults to False.

        Returns:
            DataFrame: forecast output including the following columns for each covariate.
            
            y_train: y training value at the corresponding observation for the initial window (before sliding). Values outside the initial window will be NaN, however y values will then be shown under the y_test column.
            y_test: y value (truth) corresponding to the prediction.
            y_pred: y predicted (i.e. forecast) 
            y_lower:  lower confidence limit
            y_upper:  upper confidence limit
        """
        
        assert isinstance(dfXY, pd.DataFrame)

        if isinstance(self.y,str):
            assert self.y in dfXY.columns, f' "{self.y}" column is not contained in the input DataFrame'
        else:
            for _y in self.y:
                assert _y in dfXY.columns, f' "{_y}" column is not contained in the input DataFrame'
            

 
        self.dfXYfc, self.df_pred, self.metrics, self.model, self.history_i, self.history_t, self.model_fit = sliding_forecast(dfXY, y = self.y, 
                    model=self.ml_model, model_type=self.ml_model_type, 
                    tf_params=self.tf_params, cm_params = self.cm_params,
                    covars=self.tswin_params["covars"], catvars=self.tswin_params["catvars"], 
                    exogvars=self.tswin_params["exogvars"], minmax=self.tswin_params["minmax"],
                    Nlags=self.tswin_params["Nlags"], Nobserve=self.tswin_params["Nobserve"],
                    Npred = self.tswin_params["Npred"],  Nhorizon=self.tswin_params["Nhorizon"], 
                    i_start = self.tswin_params["i_start"], idx_start=self.tswin_params["idx_start"],
                    alpha = self.tswin_params["alpha"],ci_method= self.tswin_params["ci_method"],
                    mms_cols = self.xscale["mms_cols"], ss_cols = self.xscale["ss_cols"],
                    verbose = False)
        
 
        return self.df_pred
    