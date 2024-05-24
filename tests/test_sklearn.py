import beautifulplots as bp
import pandas as pd
import sforecast as sf
from xgboost import XGBRegressor
import numpy as np
from sklearn.linear_model import LinearRegression



###### SK Learn Models

#### data Superstore
df_sales = pd.read_csv("../data/Superstore_subcatsales_2017_cdp.csv", parse_dates = ["Order Date"])
aggs = {
    "Sales":"sum",
    "Quantity":"sum"  
}

##### aggregate to category sales
df_catsales = df_sales.groupby(["Order Date" , "Category"]).agg(aggs).reset_index()

#### wide data format
dfXYw = df_catsales.copy()

def to_flat_columns(hier_cols):
    flat_cols=[]
    for clist in hier_cols:
        for n,ci in enumerate(clist):
            c = ci if n == 0 else c+"_"+ci 
        flat_cols.append(c)
    return flat_cols


dp = "Order Date" # demand period
dfXYw = dfXYw.pivot(index=dp, columns = "Category" , values = ["Quantity" , "Sales"] )
flat_cols = to_flat_columns(dfXYw.columns)
dfXYw.columns = flat_cols 
dfXYw = dfXYw.fillna(0)

def test_univariate():
    """Test Univarate Forecast"""

    y = ["Quantity_Furniture"]
    Ntest = 30
    
    dfXY = dfXYw[y]
    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    
    swin_params = {
    "Ntest":Ntest,
    "Nhorizon":1,
    "Nlags":40,
    "minmax" :(0,None)}  

    y = ["Quantity_Furniture"]

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    
    sfuv =  sf.sforecast(y = y, swin_parameters=swin_params,model=xgb_model,model_type="sk")
    df_pred_uv = sfuv.fit(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([10.50730896, 10.88909435, 17.36617851, 12.55632019,  5.52666569,
       10.25170517, 12.44235039,  9.27809048, 12.70345688,  7.40667343,
       13.36502743,  4.09783983, 15.71428585,  7.00647736, 11.07057858,
       10.12119579, 13.99941349,  9.52647591, 15.87717438,  8.08151817,
       12.39216518,  7.13974857, 13.77720547,  8.75631809, 14.66779327,
       12.57073975,  4.4809494 , 13.77108479,  3.97732449,  7.67048693])
    
    pred_expected_p = pred_expected + 5
    pred_expected_m = pred_expected - 5
    pred_result = df_pred_uv[y_pred].tail(Ntest).values
    
    
    assert (pred_result > pred_expected_m).all() , "XGBoost univariate forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "XGBoost univariate forecast failed pred_result < pred_expected_p"
    
    return df_pred_uv
    
### Multivariate, Exog, Endog, Multiple Out

# engogenous derived attributes
from sklearn.base import BaseEstimator, TransformerMixin
Nr = 3
class derived_attributes_3(BaseEstimator,TransformerMixin):
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
        if len(df) == 0:
            df = self.dfmemory
            if isinstance(dfnewrows,pd.DataFrame):
                df = pd.concat([df,dfnewrows])
        self.dfmemory = df.tail(self.Nr) 
        Nr=self.Nr
        dfnew=df.copy()
        
        dfnew["Quantity_Furniture_m1_ravg"+str(Nr)] = dfnew["Quantity_Furniture_m1"].rolling(window=Nr).mean()  
        dfnew["Quantity_Furniture_m1_rstd"+str(Nr)] = dfnew["Quantity_Furniture_m1"].rolling(window=Nr).std()
        dfnew["Quantity_Office_Supplies_m1_ravg"+str(Nr)] = dfnew["Quantity_Office Supplies_m1"].rolling(window=Nr).mean()  
        dfnew["Quantity_Office_Supplies_m1_rstd"+str(Nr)] = dfnew["Quantity_Office Supplies_m1"].rolling(window=Nr).std()
        dfnew["Quantity_Technology_m1_ravg"+str(Nr)] = dfnew["Quantity_Technology_m1"].rolling(window=Nr).mean()  
        dfnew["Quantity_Technology_m1_rstd"+str(Nr)] = dfnew["Quantity_Technology_m1"].rolling(window=Nr).std()
        # dfnew=dfnew.iloc[Nr:] # do not toss out first Nr rows since they will be NA  this will be managed by sforecast
    
        Nclip = self.Nr
        return dfnew if Nout == None else dfnew.tail(Nout)
    
    def get_Nclip(self): # returns the number of initial rows are desgarded (clipped) for NaN avoidence
        return self.Nr
    
    def get_derived_attribute_names(self):
        Nr = self.Nr
        return [ "Quantity_Furniture_m1_ravg"+str(Nr), "Quantity_Furniture_m1_rstd"+str(Nr),
                "Quantity_Office_Supplies_m1_ravg"+str(Nr), "Quantity_Office_Supplies_m1_rstd"+str(Nr),
                "Quantity_Technology_m1_ravg"+str(Nr), "Quantity_Technology_m1_rstd"+str(Nr)
                ]
          
def test_multivariate_exog_endog_mout():

    dfXY = dfXYw[["Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]].copy()
    # Exogenous Variables
    dfXY["dayofweek"] = dfXY.index.dayofweek

    y = [ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]
    Ntest= 30
    Nhorizon = 5

    # sliding forecast inputs
    swin_params = {
        "Ntest":Ntest,
        "Nhorizon":Nhorizon,
        "Nlags":40,
        "minmax" :(0,None),
        "covars":[ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"],
        "exogvars":"dayofweek",
        "derived_attributes_transform":derived_attributes_3 # Endogenous Variables
        } 

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 

    # sliding forecast model and forecast
    sfxgbmv = sf.sforecast(y = y, swin_parameters=swin_params,model=xgb_model,model_type="sk")
    df_pred_xgbmv = sfxgbmv.fit(dfXY)
    
    ypred = y[0]+"_pred"
    print("ypred =",ypred)
    print("ypred Quantity_Furniture =",df_pred_xgbmv[ypred].tail(Ntest).values)

    pred_expected_0 = [11.52530098, 16.7564106 , 11.82250023, 13.62115383,  9.6766119 ,
        1.94293904, 16.55559921, 14.6695385 , 11.22798347, 12.79857922,
       13.68799591, 12.37104416,  5.07049274, 16.40459824, 21.25341415,
        9.67630959, 16.09016991, 13.36170673, 16.59474945,  2.9206357 ,
       11.97374344, 10.58117962, 11.01480961,  8.10165024, 15.9510603 ,
        6.25594616,  7.7132287 , 12.5488615 , 16.64418602, 10.80869389]
    
    pred_expected_1=[48.81955338, 33.40526962, 33.22610474, 41.23186111, 22.25686646,
        5.31832409, 22.29068565, 32.99081802, 43.37825012, 34.12310028,
       33.30134201, 40.55511093, 25.5574894 , 30.53362465, 48.47904587,
       39.52342606, 33.73281097, 31.93192863, 22.68471146,  9.0923481 ,
       35.49350739, 52.86450577, 30.79811478, 38.94551849, 26.10488129,
       14.25063229,  7.44919491, 27.50846291, 41.53798294, 33.20026779]

    pred_expected_2 =[12.61738586,  6.37293434,  6.79109955, 17.40951347,  5.84331656,
       11.45158672, 13.69954681, 16.08982277,  9.54298401,  8.38537216,
       16.67468262,  7.20942163,  2.02668023, 16.23996162, 13.46222973,
       12.08208847, 10.02855968, 12.78578186,  2.79590511,  3.12660122,
       10.06740284, 10.94800568, 15.45762157,  7.78206253,  9.45131302,
        0.70043564,  0.        , 11.54092216,  6.6759243 ,  7.54367447]


    pred_expected_list = np.array([ pred_expected_0, pred_expected_1, pred_expected_2 ])

    print("y =",y)
    for n,_y in enumerate(y):

        _y_pred = _y+"_pred"
        print(_y_pred)
        
        pred_result = df_pred_xgbmv[_y_pred].tail(Ntest).values
        print("pred_result =", list(pred_result))

        pred_expected = pred_expected_list[n]
        print("pred_expected =",pred_expected)
        pred_expected_p = pred_expected + 50
        pred_expected_m = pred_expected - 50

        assert (pred_result > pred_expected_m).all() , f'y = {_y_pred} multivariate, exogenous, endogenous, forecast failed pred_result > pred_expected_m'
        assert (pred_result < pred_expected_p).all() , f'y = {_y_pred} multivariate, exogenous, endogenous, forecast pred_result < pred_expected_p'
        
    return df_pred_xgbmv 
        