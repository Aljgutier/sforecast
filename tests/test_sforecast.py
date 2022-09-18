import beautifulplots as bp
import pandas as pd
import sforecast as sf
from xgboost import XGBRegressor
import numpy as np


###### setup the data to be used in the next few tests ####

df_sales = pd.read_csv("./data/Superstore_subcatsales_2017_cdp.csv", parse_dates = ["Order Date"])
aggs = {
    "Sales":"sum",
    "Quantity":"sum"  
}

df_catsales = df_sales.groupby(["Order Date" , "Category"]).agg(aggs).reset_index()

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
    Npred = 30
    
    dfXY = dfXYw[y]
    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    ts_params = {
    "Npred":Npred,
    "Nhorizon":1,
    "Nlag":40,
    "minmax" :(0,None)}  

    y = ["Quantity_Furniture"]

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    
    sfuv = sf.sforecast(y = y, ts_parameters=ts_params,model=xgb_model)
    df_pred_uv = sfuv.forecast(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([11.14825916, 16.47620201, 15.51670933, 14.10373211, 8.29227543 ,14.59656048,
     15.01964188, 12.68975925, 19.6415062,   8.6744585,  12.3606863 ,  4.91769266,
     15.06090832 ,10.46361732 ,13.13412762,  6.82626009 ,17.09865952,  7.37598896,
     15.51855278 , 8.73959637 ,16.92628479 , 8.40623951, 10.60089779 ,13.41072845,
      8.79228687 , 7.79535913 , 5.81936169, 10.80079079 , 6.48547268 , 8.45924091])
    
    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_uv[y_pred].tail(Npred).values
    
    assert (pred_result > pred_expected_m).all() , "univariate forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "univariate forecast failed pred_result < pred_expected_p"
    
    
def test_univariate_covariate():
    """Test Univarate Forecast"""

    y = ["Quantity_Furniture"]
    Npred = 30
    
    dfXY = dfXYw[["Quantity_Furniture","Quantity_Office Supplies", "Quantity_Technology" ]]
    y = ["Quantity_Furniture"]
    Npred = 30

    ts_params = {
        "Npred":30,
       "Nhorizon":1,
        "Nlag":40,
        "minmax" :(0,None),
        "co_vars":[ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]} 

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    sfuvc = sf.sforecast(y = y, ts_parameters=ts_params,model=xgb_model)
    df_pred_uvc = sfuvc.forecast(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([11.255541801452637, 13.590967178344727, 17.33194351196289, 18.17267608642578, 
               9.9024076461792, 7.018445014953613, 12.39219856262207, 14.338815689086914, 17.58784294128418, 
               10.145371437072754, 15.130827903747559, 13.139010429382324, 4.695201873779297, 16.284250259399414, 
               20.907461166381836, 5.7730817794799805, 10.660463333129883, 12.513736724853516, 7.995673179626465, 
               5.665019512176514, 14.63947582244873, 9.111043930053711, 7.6075334548950195, 8.155879020690918, 
               14.03388786315918, 6.055835247039795, 2.9899139404296875, 18.785179138183594, 19.470705032348633, 
               9.454106330871582])


    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_uvc["Quantity_Furniture_pred"].tail(Npred).values


    assert (pred_result > pred_expected_m).all() , "univariate forecast with covariates failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "univariate forecast with covariates failed pred_result < pred_expected_p"
    
def test_multivariate():
    """Test Univarate Forecast"""


    
    dfXY = dfXYw[["Quantity_Furniture","Quantity_Office Supplies", "Quantity_Technology" ]]
    y = [ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]
    Npred = 30

    ts_params = {
        "Npred":Npred,
        "Nhorizon":1,
        "Nlag":40,
        "minmax" :(0,None),
        "co_vars":[ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]} 

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    sfuvc = sf.sforecast(y = y, ts_parameters=ts_params,model=xgb_model)
    df_pred_mv = sfuvc.forecast(dfXY)
    

    pred_expected_0 = [11.255541801452637, 13.590967178344727, 17.33194351196289, 18.17267608642578, 
                   9.9024076461792, 7.018445014953613, 12.39219856262207, 14.338815689086914, 
                   17.58784294128418, 10.145371437072754, 15.130827903747559, 13.139010429382324, 
                   4.695201873779297, 16.284250259399414, 20.907461166381836, 5.7730817794799805, 
                   10.660463333129883, 12.513736724853516, 7.995673179626465, 5.665019512176514, 
                   14.63947582244873, 9.111043930053711, 7.6075334548950195, 8.155879020690918, 
                   14.03388786315918, 6.055835247039795, 2.9899139404296875, 18.785179138183594, 
                   19.470705032348633, 9.454106330871582]


    pred_expected_1=[45.76167678833008, 16.458377838134766, 42.92385482788086, 41.52473449707031, 
                 31.995037078857422, 14.086807250976562, 30.055034637451172, 50.77397537231445, 
                 39.24284744262695, 30.68844985961914, 31.715389251708984, 36.315452575683594, 
                 9.167171478271484, 33.4739990234375, 40.05677795410156, 31.283485412597656, 
                 38.936798095703125, 29.54806137084961, 16.21041488647461, 13.883440017700195, 
                 36.58510971069336, 40.17214584350586, 25.843994140625, 21.877744674682617, 
                 27.117177963256836, 30.671810150146484, 4.507527828216553, 30.113576889038086, 
                 29.2062931060791, 36.58026885986328]

    pred_expected_2 = np.array([16.838611602783203, 8.737680435180664, 9.872063636779785, 
             18.513700485229492, 11.806622505187988, 13.020414352416992, 12.018507957458496, 
            11.148112297058105, 10.787961959838867, 11.249581336975098, 11.84135627746582, 
            7.271737098693848, 3.3275129795074463, 18.205322265625, 17.844205856323242, 15.652249336242676, 
            11.320643424987793, 15.34378433227539, 2.9382543563842773, 3.6095008850097656, 6.671794891357422, 
            9.964138984680176, 10.439900398254395, 12.74950885772705, 5.510839462280273, 12.97891902923584, 
            4.494198322296143, 12.835867881774902, 6.102931499481201, 13.329137802124023])



    pred_expected_list = np.array([ pred_expected_0, pred_expected_1, pred_expected_2 ])


    print("y =",y)
    for n,_y in enumerate(y):

        _y_pred = _y+"_pred"
        print(_y_pred)
        
        pred_result = df_pred_mv[_y_pred].tail(30).values
        print("pred_result =", list(pred_result))

        pred_expected = pred_expected_list[n]
        print("pred_expected =",pred_expected)
        pred_expected_p = pred_expected + 0.2
        pred_expected_m = pred_expected - 0.2
        pred_result = df_pred_mv[_y_pred].tail(30).values


        assert (pred_result > pred_expected_m).all() , f'y = {_y_pred} multivariate forecast with covariates failed pred_result > pred_expected_m'
        assert (pred_result < pred_expected_p).all() , f'y = {_y_pred} multivariate forecast with covariates failed pred_result < pred_expected_p'



