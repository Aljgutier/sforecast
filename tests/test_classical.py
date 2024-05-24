import beautifulplots as bp
import pandas as pd
import sforecast as sf
import numpy as np
from datetime import datetime

###### ARIMA 
# data shampoo
def dateparser(x):
    return datetime.strptime('190'+x, '%Y-%m')
df_shampoo = pd.read_csv("../data/shampoo.csv", parse_dates = ["Month"], date_parser=dateparser)

def test_arima():
    Ntest = 5
    y=["Sales"]
    y_pred = y[0]+"_pred"
    
    dfXY = df_shampoo[y]
    #print('dfXY')
    #print(dfXY.head())

    swin_params = {
        "Ntest":Ntest,
        "Nlags": 5,
        "minmax" :(0,None)}  

    cm_parameters = {
        "model":"arima",
        "order":(2,1,0)
    }

    sf_arima = sf.sforecast(y = y, model=None,model_type="cm", cm_parameters=cm_parameters,
                        swin_parameters=swin_params,)

    df_pred_arima = sf_arima.fit(dfXY)
    #print(df_pred_arima.tail(Ntest))
    
    pred_expected = np.array([467.811682, 519.261277, 464.182016, 615.984739, 524.253124])
    
    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_arima[y_pred].tail(Ntest).values
    
    assert (pred_result > pred_expected_m).all() , "ARIMA forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "ARIMA forecast failed pred_result < pred_expected_p"
    
    
    return df_pred_arima  # return not required for pytest ... 
 
#### SARIMAX 
# data air passengers    
df_airp = pd.read_csv("../data/AirPassengers.csv", parse_dates = ["Month"]).set_index("Month")
# add exogenous variables ... rolling nmean, rolling std, month no.
  
def test_sarimax():
    Ntest=5
    y = ["Passengers"]
    dfXY = df_airp[y]
    y_pred = y[0]+"_pred"


    swin_parameters = {
        "Ntest":Ntest,
        "Nlags":5,
        "minmax" :(0,None),
        "Nhorizon":1,
        }

    cm_parameters = {
        "model":"sarimax",
        "order":(2,1,0),
        "seasonal_order":(0,1,0,12)
        }

    sf_sarimax = sf.sforecast(y = y, model=None, model_type="cm", cm_parameters=cm_parameters,
                        swin_parameters=swin_parameters,)

    df_pred_sarimax = sf_sarimax.fit(dfXY)
    dfXY_pred_sarimax = dfXY.join(df_pred_sarimax)
    
    ts_period = pd.DateOffset(months=1)
    
    df_pred_sarimax=sf_sarimax.predict(Nperiods=1,ts_period=ts_period)
    
    pred_expected = np.array([444])

    pred_expected_p = pred_expected + 20
    pred_expected_m = pred_expected - 20
    pred_result = df_pred_sarimax[y_pred].tail(Ntest).values
    
    assert (pred_result > pred_expected_m).all() , "SARIMAX forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "SARIMAX forecast failed pred_result < pred_expected_p"

    return df_pred_sarimax
 
 
### SARIMAX w/ exogenous and endogenous
 
from sklearn.base import BaseEstimator, TransformerMixin

Nr = 3
class derived_attributes(BaseEstimator,TransformerMixin):
    def __init__(self, Nr = Nr): 
        self.Nr = Nr # slidig/rolling window rows
        self.dfmemory = None
    
    def fit(self,df):
        # ensure dataframe has enough rows
        self.dfmemory = df.tail(self.Nr) if df.index.size > self.Nr else df.index.size
        return self
    
    def transform(self,df=pd.DataFrame(), Nout=None, dfnewrows=None):
        # if df not spefified then transform on dfmemory
        # add new row(s) ... these will be provided from the predict operation
        if len(df)==0:
            df = self.dfmemory
            if isinstance(dfnewrows,pd.DataFrame):
                df = pd.concat([df,dfnewrows])
        self.dfmemory = df.tail(self.Nr) 
        Nr=self.Nr
        dfnew=df.copy()
        
        dfnew["Passengers_m1_ravg"+str(Nr)] = dfnew["Passengers_m1"].rolling(window=Nr).mean()  
        dfnew["Passengers_m1_rstd"+str(Nr)] = dfnew["Passengers_m1"].rolling(window=Nr).std()
        # dfnew=dfnew.iloc[Nr:] # do not toss out first Nr rows since they will be NA  this will be managed by sforecast
    
        Nclip = self.Nr
        return dfnew if Nout == None else dfnew.tail(Nout)
    
    def get_Nclip(self): # returns the number of initial rows are desgarded (clipped) for NaN avoidence
        return self.Nr
    
    def get_derived_attribute_names(self):
        Nr = self.Nr
        return [ "Passengers_m1_ravg"+str(Nr), "Passengers_m1_rstd"+str(Nr) ]
 
 
def test_sarimax_exog_endog():
    # Exogenous variables
    dfXY = df_airp.copy()
    dfXY["month_no"] = dfXY.index.month # exog variable

    exogvars = ["month_no"]
    
    Ntest = 5

    swin_parameters = {
        "Ntest":Ntest,
        "Nlags":5,
        "minmax" :(0,None),
        "Nhorizon":1,
        "exogvars": exogvars,
        "derived_attributes_transform":derived_attributes
        }

    cm_parameters = {
        "model":"sarimax",
        "order":(2,1,0),
        "seasonal_order":(0,1,0,12)
        }

    y = ["Passengers"]
    sf_sarimax = sf.sforecast(y = y, model=None, model_type="cm", cm_parameters=cm_parameters,
                        swin_parameters=swin_parameters,)

    df_pred_sarimax = sf_sarimax.fit(dfXY)
 
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([630, 519, 452, 413, 441])

    pred_expected_p = pred_expected + 20
    pred_expected_m = pred_expected - 20
    pred_result = df_pred_sarimax[y_pred].tail(Ntest).values
    
    assert (pred_result > pred_expected_m).all() , "SARIMAX forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "SARIMAX forecast failed pred_result < pred_expected_p"
    
    return df_pred_sarimax
 
#### AUTO ARIMA
def test_autoarima():
    Ntest = 4
    Nhorizon = 2
    dfXY = df_airp

    swin_params = {
        "Ntest":Ntest,
        "Nlags":5,
        "Nhorizon": Nhorizon,
        "minmax" :(0,None)
        }  

    cm_parameters = {
        "model":"auto_arima",
        "d":None, # let the auto search determine d
        "start_p":1,
        "start_q":1,
        "seasonal":True ,
        "D":None, # let auto search determine D
        "m":12, # 12, period (i.e., month) seasonality period
        "start_P":1,
        "start_Q":1,
        "error_action":"ignore", # don't want to know if order does not work
        "suppress_warnings":True, # don't want convergence warnings
        "stepwise":True # stepwise search
    }

    y = ["Passengers"]
    sf_autoarima = sf.sforecast(y = y, model=None, model_type="cm", cm_parameters=cm_parameters,
                        swin_parameters=swin_params,)

    df_pred_autoarima = sf_autoarima.fit(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([ 514.690165, 455.350909, 414.046232, 	457.046232	 ])

    pred_expected_p = pred_expected + 50
    pred_expected_m = pred_expected - 50
    pred_result = df_pred_autoarima[y_pred].tail(Ntest).values
    

    assert (pred_result > pred_expected_m).all() , "AUTOARIMA forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "AUTOARIMAA forecast failed pred_result < pred_expected_p"
    
    return df_pred_autoarima
 