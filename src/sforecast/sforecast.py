
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import t
from sklearn.base import clone
import numpy as np
import pandas as pd

def forecast_ci_percentile(errors, alpha, method="linear"):
    

    percentile_upper = 100 * (alpha / 2 )
    percentile_lower = 100 - 100 * (alpha / 2 )
    
    error_lower = np.percentile(errors,percentile_upper)
    error_upper = np.percentile(errors,percentile_lower)
    
    return error_lower, error_upper

def forecast_ci_tdistribution(errors, alpha):
    
    n = len(errors)
    s = np.std(errors, ddof = 1)  # sample std dev, divisor= N-ddof, delta degrees of freedom ... = N-1 for sample std dev
    t_critical = t.ppf(1-alpha/2, df = n-1) # account for both tails, prob of each tail is alpha/2
    e_mean = errors.mean()
    SE = t_critical * s / np.sqrt(n)
    
    error_lower = e_mean - SE
    error_upper = e_mean + SE

    
    return error_lower, error_upper

def forcast_confidence(df,alpha,Nhorizon = 1, error="error", method="linear", verbose=False):
    
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
            error_lower, error_upper = forecast_ci_tdistribution(errors_nh, alpha)
        elif method == "minmax":
            error_lower,error_upper = errors_nh.min(), errors_nh.max()
        elif method in percentile_methods:
             error_lower,error_upper =  forecast_ci_percentile(errors_nh, alpha, method=method)
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

def forecast_dataset(df,ar_vars, N_lags):
    # ar_vars := auto regressive varibles 
    
    dfXY = df.copy()
    
    # ensure that ar_vars is iterable
    ar_vars = [ar_vars] if isinstance(ar_vars,str) else ar_vars

    for n in np.arange(1,N_lags+1):
        arv_shift_vars = []
        for arv in ar_vars:
            arv_shift_var = arv+"_m"+str(n)
            arv_shift_vars.append(arv_shift_var)
        dfXY[arv_shift_vars] = dfXY[ar_vars].shift(n)
        
    return dfXY

def min_func(x,minvalue):
    if  ~np.isnan(x):
        x = x if x > minvalue else minvalue
    return x

def max_func(x,maxvalue):
    if ~np.isnan(x):
        x = x if x < maxvalue else maxvalue
    return x

min_vfunc = np.vectorize(min_func)
max_vfunc = np.vectorize(max_func)

def sliding_forecast(dfXY, y, model, ar_vars=None, minmax=(None,None),
                     Nlag=1, Nobserve=None, Npred = 1,  Nhorizon=1, i_start = None, 
                     idx_start=None, alpha = 0.2, ci_method="linear",verbose = False):
    
    """Receives as input DataFrame of X (exogenous + co-vvariates)  and Y (univariate or multivariate), and an untrained model.

    Args:
        ar_vars (single variable or list of variable names): dictionar containint plot options. Each key, value pair corresponds to a plot parameter
        minmax (tuple): afdsfdsaf
        Nlag: sdfdsfsf
        Nobserve: ds;lfjdsl;fj
        Npred: dfdsfjdsf
        Nhorizon: a;dlfsd;alfjsdal
        i_start: a;lkfdjldsfkj
        idx_start: a;lfsd;lfj;lsdfj
        alpha: ad;fjsd;f
        ci_method: a;dlfjds;lfkj
        verbose: a;dlfsd;lfj

    Returns:
        dfXYfc: returns None if the function completes without errors.
        df_pred: 
        metrics: 
        m: aldjf;sdfjls;afj
        
    """ 
    
    # if y not in ar_vars if not add to ar_vars
    # ensure y is iterable
    # ensure _ar_vars is iterable
    y = [y] if isinstance(y,str) else y
    if ar_vars == None: ar_vars = []
    ar_vars = [ar_vars] if isinstance(ar_vars,str) else ar_vars
    for _y in  y: 
        if _y not in ar_vars: ar_vars.append(_y)   
    
        
    # ensure minmax is iterable ... same length as y ... if there is only one minmax pair then duplicate
    if isinstance(minmax,tuple): minmax = [minmax]
    _minmax = minmax.copy()
    _minmax = len(y)*_minmax if (len(_minmax) != len(y)) and (len(_minmax)==1) else _minmax

    # assertions
    assert len(y)==len(_minmax), f'length of minmax tuples = {len(minmax)}, should equal the number of y targets'
    
    # copy the input dataframe to ensure the original is not modified
    dfXYfc = dfXY.copy()

    # auto-regressive lags
    if Nlag > 0:
        dfXYfc = forecast_dataset(dfXY,ar_vars, Nlag)
        dfXYfc = dfXYfc.iloc[Nlag:].copy()  # delete the first Nlag rows since they will contain NaNs
    
    # sliding window variables
    N = dfXYfc.index.size # total observations
    if verbose == True:
        print(f'N = {N} total observations, i-th last observation = {N-1}\n')


    # remove target variables from training variables
    
    #for _y in y:
    #    X_columns.remove(_y)
    
    # idx_start takes precedence over i_start
    # initial = initial prediction
    i_initial = N - Npred if i_start == None else i_start
    i_initial = (dfXYfc.loc[:idx_start].index.size)-1 if idx_start != None else i_initial  # i_nitial starts at 0, subtract 1
    
    
    ##### Prediction Looop ######
    metrics={}
    #### for each target ###
    df_pred=pd.DataFrame()
    X_columns = list(dfXYfc.columns)
    for n,_y in enumerate(y):
        
        _X_columns = X_columns.copy()
        # remove the target variable from X
        _X_columns.remove(_y)
 

        y_pred_values = []
        y_test_values = []
        y_pred_idx = []
        ypred_col = _y +"_pred"
        ytrain_col = _y +"_train"
        ytest_col  = _y +"_test"
        error_col = _y + "_pred_error"
        
        #### sliding/expanding window forcast  ###
        #### Nhorizon predict forward Nhorizon steps
        for i in np.arange(i_initial,i_initial+Npred,Nhorizon):  

            # if Nobserve == None then expanding window, else sliding window
            # i0 first observation
            i0 = 0 if Nobserve == None else i - Nobserve

            if verbose == True:
                print(f'training observations ilocs = {i0}:{i-1} ')
            
            X_train = dfXYfc[_X_columns].iloc[i0:i].copy().values  # last train index is i-1
            Y_train = dfXYfc[_y].iloc[i0:i].copy().values         # last train index is i-1

            m = clone(model)  # clone untrained model
            m.fit(X_train,Y_train)
            
            for j in np.arange(i,i + Nhorizon,1):
                if j == N: break
                    
                if verbose == True:
                    print(f'   prediction input iloc ={j}')
                
                X_test = dfXYfc[_X_columns].iloc[j:j+1].copy().values
                y_pred = m.predict(X_test) # prediction
                y_test = dfXYfc[_y].iloc[j]
                
                y_pred_values.append(y_pred[0])
                y_test_values.append(y_test)
                y_pred_idx.append(j)
                
                if verbose == True:
                    print(f'   y_test = {y_test}  y_pred = {y_pred[0]}')      
        

                        
        # y_pred, y_train, y_test
        # apply min max limits to the forecast
        y_pred_values = min_vfunc(y_pred_values,_minmax[n][0]) if _minmax[n][0] != None else y_pred_values
        y_pred_values = max_vfunc(y_pred_values,_minmax[n][1]) if _minmax[n][1] != None else y_pred_values
        

        errors=list(np.array(y_test_values) - np.array(y_pred_values))
        _df_pred = pd.DataFrame(index=dfXYfc.index) # keep the index from dfXYfc
        idx = dfXYfc.iloc[y_pred_idx].index # prediction indexes
        _df_pred[ytrain_col] = dfXYfc.iloc[i0:i_initial][_y] # y train
        _df_pred[ytest_col] = dfXYfc.iloc[i_initial:][_y] # y test
        _df_pred[ypred_col] = np.nan
        _df_pred.loc[idx, ypred_col] = y_pred_values # prediction values 
        _df_pred.loc[idx,error_col] = errors

        # metrics dictionary ... statistics
        y_test_array = dfXYfc[_y].iloc[i_initial:i_initial+Npred]
        y_pred_array = _df_pred.iloc[i_initial:i_initial+Npred][ypred_col]
        rmse_test = np.sqrt(MSE(y_test_array, y_pred_array))
        mae_test= MAE(y_test_array, y_pred_array )
        metrics[ypred_col] = {"RMSE":rmse_test, "MAE":mae_test}

        # confidence intervals
        _df_pred = forcast_confidence(_df_pred,alpha=alpha, error=error_col, Nhorizon=Nhorizon, method=ci_method, verbose=verbose)
        ypred_lower = ypred_col +"_lower"
        ypred_upper = ypred_col +"_upper"
        _df_pred[ypred_lower] = _df_pred[ypred_col] + _df_pred["error_lower"]
        _df_pred[ypred_upper] = _df_pred[ypred_col] + _df_pred["error_upper"]

        # Apply minmax to CIs

        _df_pred[ypred_lower] = _df_pred[ypred_lower].apply(lambda x: min_func(x,_minmax[0][0])) if _minmax[0][0]!=None else  _df_pred[ypred_lower]
        _df_pred[ypred_upper] = _df_pred[ypred_upper].apply(lambda x: max_func(x,_minmax[0][1])) if _minmax[0][1]!=None else  _df_pred[ypred_upper]
       
       # drop error_lower and error_upper
        _df_pred=_df_pred.drop(["error_lower","error_upper"],axis=1)    
        
        
        # join to wide df_pred frame
        df_pred = df_pred.join(_df_pred) if n!=0 else _df_pred

        if verbose == True:
            print(f'\nmetrics = {metrics}')
            


    return dfXYfc, df_pred, metrics, m


class sforecast:
    def __init__(self,  y="y", model=None, ts_parameters="None",):
        
        # initialize variables
        self.ts_params = {
            "ar_vars":None,
            "Nobserve":None,
            "Npred":1,
            "Nlag":1,
            "Nhorizon":1,
            "i_start":None,
            "idx_start":None,
            "alpha":0.2,
            "ci_method":"linear",
            "minmax":(None,None)
        }
        
        self.y = y
        self.model = model
        
        for k in ts_parameters:
            assert self.ts_params.__contains__(k), f'ts_parameters key  = {k}, is not valid'
            self.ts_params[k] = ts_parameters[k]
            
        assert self.y != None, f'y forecast variable(s) not specified'


    def forecast(self, dfXY, verbose=False):
        
        assert isinstance(dfXY, pd.DataFrame)
        df_forecast = pd.DataFrame()
 
        self.dfXYfc, self.df_forecast, self.metrics, self.model = sliding_forecast(dfXY, y = self.y, model=self.model, 
                    ar_vars=self.ts_params["ar_vars"], minmax=self.ts_params["minmax"],
                    Nlag=self.ts_params["Nlag"], Nobserve=self.ts_params["Nobserve"],
                    Npred = self.ts_params["Npred"],  Nhorizon=self.ts_params["Nhorizon"], 
                    i_start = self.ts_params["i_start"], idx_start=self.ts_params["idx_start"],
                    alpha = self.ts_params["alpha"],ci_method= self.ts_params["ci_method"],
                    verbose = False)
 
        return self.df_forecast
    
    

    