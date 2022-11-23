import beautifulplots as bp
import pandas as pd
import sforecast as sf
from xgboost import XGBRegressor
import numpy as np
from datetime import datetime


from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, concatenate, Dropout
from keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder




### 3 Sections
# 1. Classic Models
# 2. SK Learn Models
# 3. TensorFlow Models



###### Classic Models

# shampoo sales
def dateparser(x):
    return datetime.strptime('190'+x, '%Y-%m')
df_shampoo = pd.read_csv("./data/shampoo.csv", parse_dates = ["Month"], date_parser=dateparser)

def test_arima():
    
    Npred=5
    dfXY = df_shampoo

    tswin_params = {
        "Npred":Npred,
        "Nlags":5,
        "minmax" :(0,None)}  

    cm_parameters = {
        "model":"arima",
        "order":(2,1,0)
    }

    y = ["Sales"]
    sf_arima = sf.sforecast(y = y, model=None,model_type="cm", cm_parameters=cm_parameters,
                        tswin_parameters=tswin_params,)
    
    df_pred_arima = sf_arima.forecast(dfXY)
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([467.811682, 519.261277, 464.182016, 615.984739, 524.253124])

    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_arima[y_pred].tail(Npred).values
    
    assert (pred_result > pred_expected_m).all() , "ARIMA forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "ARIMA forecast failed pred_result < pred_expected_p"
 
# air passengers    
df_airp = pd.read_csv("./data/AirPassengers.csv", parse_dates = ["Month"]).set_index("Month")
# add exogenous variables ... rolling nmean, rolling std, month no.
Nr=12
df_airp["ravg"] = df_airp["Passengers"].rolling(window=Nr).mean()  
df_airp["rstd"] = df_airp["Passengers"].rolling(window=Nr).std()
df_airp=df_airp.iloc[Nr:] # toss out first Nr rows since they will be NA due to rolling mean and std
df_airp["month_no"] = df_airp.index.month    


def test_sarimax():
    Npred=5
    dfXY = df_airp

    tswin_params = {
        "Npred":Npred,
        "Nlags":5,
        "minmax" :(0,None)}  

    cm_parameters = {
        "model":"sarimax",
        "order":(2,1,0),
        "seasonal_order":(0,1,0,12)
    }

    y =["Passengers"]
    sf_sarimax = sf.sforecast(y = y, model=None, model_type="cm", cm_parameters=cm_parameters,
                        tswin_parameters=tswin_params,)

    df_pred_sarimax = sf_sarimax.forecast(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([630.076032, 519.160167, 452.098118, 413.103090, 441.669454])

    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_sarimax[y_pred].tail(Npred).values
    
    assert (pred_result > pred_expected_m).all() , "SARIMAX forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "SARIMAX forecast failed pred_result < pred_expected_p"
 

def test_autoarima():
    Npred=3
    dfXY = df_airp

    tswin_params = {
        "Npred":Npred,
        "Nlags":5,
        "Nhorizon":3,
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
                        tswin_parameters=tswin_params,)

    df_pred_autoarima = sf_autoarima.forecast(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([452.642947, 407.436257, 450.502702])

    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_autoarima[y_pred].tail(Npred).values
    

    assert (pred_result > pred_expected_m).all() , "AUTOARIMA forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "AUTOARIMAA forecast failed pred_result < pred_expected_p"
 
     
###### SK Learn Models
# setup the data to be used in the next few tests 

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
    
    tswin_params = {
    "Npred":Npred,
    "Nhorizon":1,
    "Nlags":40,
    "minmax" :(0,None)}  

    y = ["Quantity_Furniture"]

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    
    sfuv = sf.sforecast(y = y, tswin_parameters=tswin_params,model_type = "sk", model=xgb_model)
    df_pred_uv = sfuv.forecast(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([11.148259162902832, 16.4762020111084, 15.516709327697754, 14.103732109069824,
                    5.3794145584106445, 14.59656047821045, 15.019641876220703, 12.689759254455566, 19.64150619506836, 
                    8.674458503723145, 12.360686302185059, 4.9176926612854, 15.060908317565918, 10.463617324829102, 
                    13.134127616882324, 6.826260089874268, 17.09865951538086, 7.375988960266113, 15.518552780151367, 
                    8.739596366882324, 16.926284790039062, 8.40623950958252, 10.600897789001465, 13.410728454589844, 
                    8.79228687286377, 8.522675514221191, 5.819361686706543, 10.800790786743164, 6.485472679138184, 
                    8.459240913391113])
    
    pred_expected_p = pred_expected + 0.2
    pred_expected_m = pred_expected - 0.2
    pred_result = df_pred_uv[y_pred].tail(Npred).values
    
    assert (pred_result > pred_expected_m).all() , "XGBoost univariate forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "XGBoost univariate forecast failed pred_result < pred_expected_p"
    
    
def test_univariate_covariate():
    """Test Univarate Forecast"""

    y = ["Quantity_Furniture"]
    Npred = 30
    
    dfXY = dfXYw[["Quantity_Furniture","Quantity_Office Supplies", "Quantity_Technology" ]]
    y = ["Quantity_Furniture"]
    Npred = 30

    tswin_params = {
        "Npred":30,
       "Nhorizon":1,
        "Nlags":40,
        "minmax" :(0,None),
        "covars":[ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]} 

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    sfuvc = sf.sforecast(y = y, tswin_parameters=tswin_params,model_type = "sk", model=xgb_model)
    df_pred_uvc = sfuvc.forecast(dfXY)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([11.255541801452637, 13.590967178344727, 17.33194351196289, 18.17267608642578, 
                    9.9024076461792, 7.018445014953613, 12.39219856262207, 14.338815689086914, 17.58784294128418, 
                    10.949801445007324, 15.130827903747559, 13.139010429382324, 4.695201873779297, 16.284250259399414, 
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

    tswin_params = {
        "Npred":Npred,
        "Nhorizon":1,
        "Nlags":40,
        "minmax" :(0,None),
        "covars":[ "Quantity_Furniture", "Quantity_Office Supplies", "Quantity_Technology"]} 

    xgb_model = XGBRegressor(n_estimators = 10, seed = 42, max_depth=5) 
    sfuvc = sf.sforecast(y = y, tswin_parameters=tswin_params,  model_type="sk",  model=xgb_model)
    df_pred_mv = sfuvc.forecast(dfXY)
    

    pred_expected_0 = [11.255541801452637, 13.590967178344727, 17.33194351196289, 18.17267608642578, 9.9024076461792, 7.018445014953613, 
                       12.39219856262207, 14.338815689086914, 17.58784294128418, 10.949801445007324, 15.130827903747559, 13.139010429382324, 
                       4.695201873779297, 16.284250259399414, 20.907461166381836, 5.7730817794799805, 10.660463333129883, 12.513736724853516,
                       7.995673179626465, 5.665019512176514, 14.63947582244873, 9.111043930053711, 7.6075334548950195, 8.155879020690918, 
                       14.03388786315918, 6.055835247039795, 2.9899139404296875, 18.785179138183594, 19.470705032348633, 9.454106330871582]


    pred_expected_1=[45.76167678833008, 16.458377838134766, 42.92385482788086, 41.52473449707031, 31.995037078857422, 
                     14.086807250976562, 30.055034637451172, 50.77397537231445, 39.24284744262695, 30.68844985961914, 
                     31.715389251708984, 36.315452575683594, 9.167171478271484, 33.4739990234375, 40.05677795410156, 
                     31.283485412597656, 38.936798095703125, 29.54806137084961, 16.21041488647461, 13.883440017700195, 
                     36.58510971069336, 40.17214584350586, 25.843994140625, 21.877744674682617, 27.117177963256836, 
                     30.671810150146484, 4.507527828216553, 30.113576889038086, 29.2062931060791, 36.58026885986328]

    pred_expected_2 =[16.838611602783203, 8.737680435180664, 9.872063636779785, 18.513700485229492, 11.806622505187988, 
                      13.020414352416992, 12.018507957458496, 11.148112297058105, 10.787961959838867, 11.249581336975098, 
                      11.84135627746582, 7.271737098693848, 3.3275129795074463, 18.205322265625, 17.844205856323242, 
                      15.652249336242676, 11.320643424987793, 15.34378433227539, 2.9382543563842773, 3.6095008850097656, 
                      6.671794891357422, 9.964138984680176, 10.439900398254395, 10.336274147033691, 5.510839462280273, 
                      12.97891902923584, 4.494198322296143, 12.835867881774902, 6.102931499481201, 13.329137802124023]


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


###### TensorFlow Models

df_m5sales20 = pd.read_csv("./data/m5_sales_7_items_events_cci_wide.csv", parse_dates = ["date"])
df_m5sales20 = df_m5sales20.set_index("date")

# variable types
covars = [c for c in df_m5sales20.columns if "unit_sales_CA_1_" in c]
catvars = [ "weekday", "event_name_1","event_name_2"]
exogvars = [ "year", "month" , "week",  "snap_CA",  "CCI_USA"]
Ncatvars = len(catvars)
Ncovars = len(covars)
Nexogvars = len(exogvars)

# dfXY ... covars + exogvars + catvars
cols = covars+catvars+exogvars
dfXY = df_m5sales20[cols].copy()

# label Encoding
le_catvars = [ "le_"+c for c in catvars ] # label encoded category columns
le = LabelEncoder()
dfXY[le_catvars] =dfXY[catvars].apply(le.fit_transform)
print(f'N event_name_1 labels = {dfXY.groupby("event_name_1")["event_name_1"].count().index.size}')

# embedding dimensions
eindim = [dfXY[le_catvars].groupby(c)[c].count().index.size + 1 for c in le_catvars] # add 1 to the dim or err in TF
eoutdim = [np.rint(np.log2(x)).astype(int) for x in eindim]


def test_tf_univariate():

    # y forecast variable
    y = ["unit_sales_CA_1_FOODS_3_FOODS_3_030"]

    # univariate data
    print("dfXYm5 univariate")
    dfXYuv = dfXY[y] 
    
    # TensorFlow model, dense network, 3 hidden layers

    Nlags=5
    inputs = Input((Nlags,))
    h1 = Dense(Nlags, activation='relu')(inputs)
    h2 = Dense(20, activation='relu')(h1)
    h3 = Dense(10, activation='relu')(h2)
    output = Dense(1)(h3)
    model_tf_dense = Model(inputs=inputs, outputs=output)

    # define optimizer and compile
    optimizer = Adam(learning_rate=0.05, decay=.1)
    model_tf_dense.compile(loss='mse', optimizer=optimizer)
    print(model_tf_dense.summary())

    Npred = 30
    tswin_params = {
        "Npred":Npred,
        "Nhorizon":5,
        "Nlags":Nlags,
        "minmax" :(0,None)
        }  


    tf_params = {
        "Nepochs_i": 100,
        "Nepochs_t": 100,
        "batch_size":100
        }


    sfuvtf = sf.sforecast(y = y, model_type="tf", tswin_parameters=tswin_params,model=model_tf_dense, tf_parameters=tf_params)
    df_pred_uv = sfuvtf.forecast(dfXYuv)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([ 7.94464588,  8.53678799,  6.36485577,  7.69052935,  6.32480049,
        7.70148897,  6.55430937, 11.47074127, 10.59178066,  7.99912071,
        6.07184362,  5.82350636,  8.08796597,  7.40571404, 11.26749325,
        6.41733456,  6.94715118,  7.61641932,  6.05367947,  4.62289476,
        5.57118893,  6.66552591,  8.29196644,  6.66091871,  6.85628557,
        7.43496895,  7.4591279 ,  9.36265373,  8.71832561,  8.94917297])

    pred_expected_p = pred_expected + 2
    pred_expected_m = pred_expected - 2
    pred_result = df_pred_uv[y_pred].tail(Npred).values
    
    print(df_pred_uv[y_pred].tail(Npred).values)

    assert (pred_result > pred_expected_m).all() , "TensorFlow Univariate forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "TensorFlow Univariate forecast failed pred_result < pred_expected_p"
 
 
def test_univariate_embeddings():
     
    # y forecast variable
    y = ["unit_sales_CA_1_FOODS_3_FOODS_3_030"]
    dfXYuvcat = dfXY[y+le_catvars]
    
    # Dense Network (Xlags = univariate lags) + Embeddings (Categorical Variables)
    # TensorFlow model ... Categorical Embeddings + Dense (Continuous Variables)
    Nlags = 5
    Nconts = Nlags * Ncovars + Nexogvars # in general, lagged covars (does not include unlagged covars) + exogvars
    Ndense = Nlags  # N continous/dense variables, in this case covars is 1 (univarate)
    Nembout = sum(eoutdim)

    # Dense Network, 2 hidden layers, continuous variables ... covar lags and exogenous variables
    cont_inputs = Input((Ndense,))
    h1c = Dense(Ndense, activation='relu')(cont_inputs)


    # embeddings, cat vars

    cat_inputs_list = [ Input((1,)) for c in range(Ncatvars) ]  # one embedding for each categorical variable
    emb_out_list = [Embedding(ein,eout,input_length=1)(cat) for ein,eout,cat in zip(eindim ,eoutdim,cat_inputs_list) ]
    emb_flat_list = [Flatten()(emb_out) for emb_out in emb_out_list ]

    # combined 
    combined = concatenate([h1c]+emb_flat_list)

    # dense reduction layers
    Nh1_comb = Ndense + Nembout  # 
    h1_comb = Dense(Nh1_comb, activation='relu')(combined)
    Nh2_comb = np.rint(Nh1_comb/2).astype(int)
    h2_comb = Dense(Nh2_comb, activation='relu')(h1_comb)

    # output
    output = Dense(1)(h2_comb)  # linear activation ... linear combination 

    # build model
    model_tf_dense_emb = Model(inputs=[cont_inputs, cat_inputs_list], outputs=output)

    # define optimizer and compile ...
    optimizer = Adam(learning_rate=0.05, decay=.1)
    model_tf_dense_emb.compile(loss='mse', optimizer=optimizer)
    print(model_tf_dense_emb.summary())

    # forecast
    Npred = 2
    tswin_params = {
        "Npred":Npred,
        "Nhorizon":1,
        "Nlags":Nlags,
        "minmax" :(0,None),
        "catvars":le_catvars 
        }  

    tf_params = {
        "Nepochs_i": 500,
        "Nepochs_t": 200,
        "batch_size":100  
    }

    sfccm5 = sf.sforecast(y = y, model_type="tf", tswin_parameters=tswin_params,model=model_tf_dense_emb, tf_parameters=tf_params)

    df_pred_ccm5 = sfccm5.forecast(dfXYuvcat)

    print(f'\nmetrics = {sfccm5.metrics}')
    dfXY_pred_ccm5 = dfXY.join(df_pred_ccm5)

    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([9.04439735, 9.28969383])

    pred_expected_p = pred_expected + 2
    pred_expected_m = pred_expected - 2
    pred_result = df_pred_ccm5[y_pred].tail(Npred).values
    
    print(df_pred_ccm5[y_pred].tail(Npred).values)

    assert (pred_result > pred_expected_m).all() , "TensorFlow Univariate forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "TensorFlow Univariate forecast failed pred_result < pred_expected_p"
 
 
def test_multivariate_categoricao_singlout():
    
    # y forecast variable
    y = ["unit_sales_CA_1_FOODS_3_FOODS_3_030" ]
    dfXYmvcatso = dfXY[covars+exogvars+le_catvars]
    
    # TensorFlow model ... Categorical Embeddings + Dense (Continuous Variables), Multiple Ouput
    Nlags = 5
    Ndense = Nlags * Ncovars + Nexogvars #lagged covars (does not include unlagged covars) + exogvars
    Nemb = Ncatvars
    Nembout = sum(eoutdim)

    print(f'Ndense = {Ndense}')
    print(f'Nemb = {Nemb}')

    # Dense Network, 2 hidden layers, continuous variables ... covar lags and exogenous variables
    cont_inputs = Input((Ndense,))
    h1d = Dense(Ndense, activation='relu')(cont_inputs)

    # embeddings, cat vars
    cat_inputs_list = [ Input((1,)) for c in range(Nemb) ]  # one embedding for each categorical variable
    emb_out_list = [Embedding(ein,eout,input_length=1)(cat) for ein,eout,cat in zip(eindim ,eoutdim,cat_inputs_list) ]
    emb_flat_list = [Flatten()(emb_out) for emb_out in emb_out_list ]

    # combined 
    combined = concatenate([h1d]+emb_flat_list)
    combined_d = Dropout(0.2)(combined)

    # dense reduction layers
    Nh1c = Ndense + Nembout # 
    h1c = Dense(Nh1c, activation='relu')(combined_d)
    h1c_d = Dropout(0.2)(h1c)
    Nh2c = np.rint(Nh1c/2).astype(int)
    h2c = Dense(Nh2c, activation='relu')(h1c_d)
    h2c_d = Dropout(0.2)(h2c)

    # output
    output = Dense(1)(h2c_d)  # linear activation ... linear combination 
    model_tf_dense_emb_so = Model(inputs=[cont_inputs, cat_inputs_list], outputs=output)

    # define optimizer and compile
    optimizer = Adam(learning_rate=0.07, decay=.2)
    model_tf_dense_emb_so.compile(loss='mse', optimizer=optimizer)

    Npred = 3
    tswin_params = {
        "Npred":Npred,
        "Nhorizon":1,
        "Nlags":Nlags,
        "minmax" :(0,None),
        "covars":covars,
        "catvars":le_catvars

        }  

    tf_params = {
        "Nepochs_i": 500,
        "Nepochs_t": 200,
        "batch_size":100  
    }

    sfcovembso = sf.sforecast(y = y, model_type="tf", tswin_parameters=tswin_params,model=model_tf_dense_emb_so, tf_parameters=tf_params)
    
    df_pred_mvcatso = sfcovembso.forecast(dfXYmvcatso)

    print(f'\nmetrics = {sfcovembso.metrics}')
    dfXY_pred_mvcatso = dfXY.join(df_pred_mvcatso)
    
    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([9.36774826, 7.46592999, 7.43217564])

    pred_expected_p = pred_expected + 2
    pred_expected_m = pred_expected - 2
    pred_result = df_pred_mvcatso[y_pred].tail(Npred).values
    
    print(df_pred_mvcatso[y_pred].tail(Npred).values)

    assert (pred_result > pred_expected_m).all() , "TensorFlow multivariate-categorical-single-output forecast failed pred_result > pred_expected_m"
    assert (pred_result < pred_expected_p).all() , "TensorFlow multivariate-categorical-single-output forecast failed pred_result < pred_expected_p"
 
 
def test_multivariate_categoricao_multiout():
    
    y = ["unit_sales_CA_1_FOODS_3_FOODS_3_030", "unit_sales_CA_1_FOODS_2_FOODS_2_044" ]
    dfXYmvcatmo = dfXY[covars+exogvars+le_catvars]
    
    # TensorFlow Model
    # Dense (exog + covariate lags) + Embeddings (categorical variables)
    
    Nlags = 5
    Ndense = Nlags * Ncovars + Nexogvars #lagged covars (does not include unlagged covars) + exogvars
    Nemb = Ncatvars
    Nembout = sum(eoutdim)
    Nout = len(y)

    print(f'Ndense = {Ndense}')
    print(f'Nemb = {Nemb}')
    print(f'Nout = {Nout}')

    # Dense Network, 2 hidden layers, continuous variables ... covar lags and exogenous variables
    cont_inputs = Input((Ndense,))
    h1d = Dense(Ndense, activation='relu')(cont_inputs)

    # embeddings, cat vars
    cat_inputs_list = [ Input((1,)) for c in range(Nemb) ]  # one embedding for each categorical variable
    emb_out_list = [Embedding(ein,eout,input_length=1)(cat) for ein,eout,cat in zip(eindim ,eoutdim,cat_inputs_list) ]
    emb_flat_list = [Flatten()(emb_out) for emb_out in emb_out_list ]

    # combined 
    combined = concatenate([h1d]+emb_flat_list)
    combined_d = Dropout(0.2)(combined)

    # dense reduction layers
    Nh1c = Ndense + Nembout # 
    h1c = Dense(Nh1c, activation='relu')(combined_d)
    h1c_d = Dropout(0.2)(h1c)
    Nh2c = np.rint(Nh1c/2).astype(int)
    h2c = Dense(Nh2c, activation='relu')(h1c_d)
    h2c_d = Dropout(0.2)(h2c)

    # output
    output = Dense(Nout)(h2c_d)  # linear activation ... linear combination 
    model_tf_dense_emb_mo = Model(inputs=[cont_inputs, cat_inputs_list], outputs=output)
    
    
    # Forecast
    Npred = 3
    tswin_params = {
        "Npred":Npred,
        "Nhorizon":1,
        "Nlags":Nlags,
        "minmax" :(0,None),
        "covars":covars,
        "catvars":le_catvars

        }  

    tf_params = {
        "Nepochs_i": 500,
        "Nepochs_t": 200,
        "batch_size":100  
    }

    # define optimizer and compile ...compile before calling sf forcast or will start with pre-trained weights
    optimizer = Adam(learning_rate=0.07, decay=.2)
    model_tf_dense_emb_mo.compile(loss='mse', optimizer=optimizer)

    sfmvembmo = sf.sforecast(y = y, model_type="tf", tswin_parameters=tswin_params,model=model_tf_dense_emb_mo, tf_parameters=tf_params)

    df_pred_mvcatmo = sfmvembmo.forecast(dfXYmvcatmo)

    print(f'\nmetrics = {sfmvembmo.metrics}')
    dfXY_pred_mvcatmo = dfXY.join(df_pred_mvcatmo)
    
   

    
    pred_expected_0 =  [9.54304504 , 8.80074978 , 6.8367238 ]
    pred_expected_1 = [0.79318213 , 0.79663813 , 0.80142403]

    pred_expected_list = np.array([ pred_expected_0, pred_expected_1 ])
    
    print("y =",y)
    for n,_y in enumerate(y):
        _y_pred = _y+"_pred"
        print(_y_pred)
        
        pred_result = df_pred_mvcatmo[_y_pred].tail(Npred).values
        print("pred_result =", list(pred_result))

        pred_expected = pred_expected_list[n]
        print("pred_expected =",pred_expected)
        pred_expected_p = pred_expected + 2
        pred_expected_m = pred_expected - 2
        pred_result = df_pred_mvcatmo[_y_pred].tail(Npred).values

        assert (pred_result > pred_expected_m).all() , f'y = {_y_pred} TensorFlow multivariate-categorical-multi-output {n} forecast with covariates failed pred_result > pred_expected_m'
        assert (pred_result < pred_expected_p).all() , f'y = {_y_pred} TensorFlow multivariate-categorical-multi-output {n} forecast with covariates failed pred_result < pred_expected_p'


def test_multivariate_categoricao_multiout():
    
    y = ["unit_sales_CA_1_FOODS_3_FOODS_3_030", "unit_sales_CA_1_FOODS_2_FOODS_2_044" ]
    dfXYmvexogcatmo = dfXY[covars+exogvars+le_catvars]
    
    # TensorFlow
    # LSTM (covariate lags) + Dense (exogenous vars) + Embeddings (categorical vars)
    
    # TensorFlow model ... LSTM (covar lags) Embeddings (categorical) + Dense (exog vars), Multiple Ouput
    Nlags = 5
    Nlstmsteps = Nlags   #lagged covars (does not include unlagged covars)
    Nlstmfeatures = Ncovars
    Ndense = Nexogvars # exogvars
    Nemb = Ncatvars
    Nembout = sum(eoutdim)
    Nout = len(y)

    print("Nlstmsteps =",Nlstmsteps)
    print("Nlstmfeatures =",Ncovars)
    print(f'Ndense = {Ndense}')
    print(f'Nemb = {Nemb}')
    print(f'Nout = {Nout}')

    lstm_inputs = Input((Nlstmsteps,Nlstmfeatures))   # number of inputs = Nlags , number of features = Ncovars
    h1lstm= LSTM(Nlstmsteps, activation='relu', input_shape = (Nlstmsteps, Nlstmfeatures))(lstm_inputs) # nsteps = Nlags

    # Dense Network, 2 hidden layers, continuous variables ... covar lags and exogenous variables
    dense_inputs = Input((Ndense,))
    h1d = Dense(Ndense, activation='relu')(dense_inputs)

    # embeddings, cat vars
    emb_inputs_list = [ Input((1,)) for c in range(Nemb) ]  # one embedding for each categorical variable
    emb_out_list = [Embedding(ein,eout,input_length=1)(cat) for ein,eout,cat in zip(eindim ,eoutdim,emb_inputs_list) ]
    emb_flat_list = [Flatten()(emb_out) for emb_out in emb_out_list ]

    # combined 
    combined = concatenate([h1lstm] +[h1d]+emb_flat_list)
    combined_d = Dropout(0.2)(combined)

    # dense reduction layers
    Nh1c = Nlstmsteps*Nlstmfeatures + Ndense + Nembout # 
    h1c = Dense(Nh1c, activation='relu')(combined_d)
    h1c_d = Dropout(0.2)(h1c)
    Nh2c = np.rint(Nh1c/2).astype(int)
    h2c = Dense(Nh2c, activation='relu')(h1c_d)
    h2c_d = Dropout(0.2)(h2c)

    # output
    output = Dense(Nout)(h2c_d)  # linear activation ... linear combination 
    model_tf_lstm_dense_emb_mo = Model(inputs=[lstm_inputs, dense_inputs, emb_inputs_list], outputs=output)

    # define optimizer and compile ..
    optimizer = Adam(learning_rate=0.07, decay=.2)
    model_tf_lstm_dense_emb_mo.compile(loss='mse', optimizer=optimizer)
    
    Npred = 2 
    tswin_params = {
        "Npred":Npred,
        "Nhorizon":1,
        "Nlags":Nlags,
        "minmax" :(0,None),
        "covars":covars,
        "catvars":le_catvars,
        "exogvars":exogvars
        }  

    tf_params = {
        "Nepochs_i": 1000,
        "Nepochs_t": 400,
        "batch_size":100,
        "lstm": True
    }

    sfmvexogembmo = sf.sforecast(y = y, model_type="tf", tswin_parameters=tswin_params,model=model_tf_lstm_dense_emb_mo, tf_parameters=tf_params)

    df_pred_mvexogcatmo =sfmvexogembmo.forecast(dfXYmvexogcatmo)

    print(f'\nmetrics = {sfmvexogembmo.metrics}')
    dfXY_pred_covexogcatmo = dfXY.join(df_pred_mvexogcatmo)
    
    
        
    pred_expected_0 = [6.24292707, 6.28484058]
    pred_expected_1 = [0.79908311 ,0.80192828]

    pred_expected_list = np.array([ pred_expected_0, pred_expected_1 ])
    
    print("y =",y)
    for n,_y in enumerate(y):
        _y_pred = _y+"_pred"
        print(_y_pred)
        
        pred_result = df_pred_mvexogcatmo [_y_pred].tail(Npred).values
        print("pred_result =", list(pred_result))

        pred_expected = pred_expected_list[n]
        print("pred_expected =",pred_expected)
        pred_expected_p = pred_expected + 2
        pred_expected_m = pred_expected - 2
        pred_result = df_pred_mvexogcatmo[_y_pred].tail(Npred).values

        assert (pred_result > pred_expected_m).all() , f'y = {_y_pred} TensorFlow multivariate-categorical-multi-output {n} forecast with covariates failed pred_result > pred_expected_m'
        assert (pred_result < pred_expected_p).all() , f'y = {_y_pred} TensorFlow multivariate-categorical-multi-output {n} forecast with covariates failed pred_result < pred_expected_p'



