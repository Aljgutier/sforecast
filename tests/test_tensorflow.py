
# sforecast
import sforecast as sf
print(f'sforecast version = {sf.__version__}')

# python - pandas
import pandas as pd
import numpy as np
from datetime import datetime

from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, concatenate, Dropout
from keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder




##### data ... M5 Sales, 7 Items ... Daily
df_m5sales7 = pd.read_csv("../data/m5_sales_7_items_events_cci_wide.csv", parse_dates = ["date"])
df_m5sales7 = df_m5sales7.set_index("date")

# variable types
covars = [c for c in df_m5sales7.columns if "unit_sales_CA1_" in c]
catvars = [ "weekday", "event_name_1","event_name_2"]
exogvars = [ "year", "month" , "week",  "snap_CA",  "CCI_USA"]
Ncatvars = len(catvars)
Ncovars = len(covars)
Nexogvars = len(exogvars)

# dfXY ... covars + exogvars + catvars
cols = covars+catvars+exogvars
dfXY = df_m5sales7[cols].copy()

# label Encoding
le_catvars = [ "le_"+c for c in ["event_name_1","event_name_2"] ] # label encoded category columns ... weekday already encoded
print(le_catvars)
le = LabelEncoder()
dfXY[le_catvars] =dfXY[["event_name_1","event_name_2"] ].apply(le.fit_transform)
weekday_num = {"Sunday":0, "Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6}
dfXY["le_weekday"] = dfXY["weekday"].apply( lambda x: weekday_num[x]) # create our own labels or label_encoder creates arbitrary number assignments
le_catvars = ["le_weekday"] + le_catvars # weekday is alread encoded ... add to le_catvars 
print(f'le_catvars = {le_catvars}')
print(f'N event_name_1 labels = {dfXY.groupby("event_name_1")["event_name_1"].count().index.size}')

# embedding dimensions
eindim = [dfXY[le_catvars].groupby(c)[c].count().index.size + 1 for c in le_catvars] # add 1 to the dim or err in TF
eoutdim = [np.rint(np.log2(x)).astype(int) for x in eindim]
print(f'eindim = {eindim}')
print(f'eoutdim = {eoutdim}')



def test_univariate(_assert=True):
    
    # y forecast variable

    y = ["unit_sales_CA1_FOODS_030"]

    # univariate data
    print("dfXYtf univariate")
    dfXYtf = dfXY[y] 

    # display data
    
    # Neural Network Initialization
    Nlags=5
    inputs = Input((Nlags,))
    h1 = Dense(Nlags, activation='relu')(inputs)
    h2 = Dense(20, activation='relu')(h1)
    h3 = Dense(10, activation='relu')(h2)
    output = Dense(1)(h3) # linear output
    model_tf_dense = Model(inputs=inputs, outputs=output)

    # define optimizer and compile
    optimizer = Adam(learning_rate=0.05, decay=.1)
    model_tf_dense.compile(loss='mse', optimizer=optimizer)
    print(model_tf_dense.summary())
        
    
    # Model Initialization - WARNING
    # ... recreate and compile the model ... reruning this cell will tune the existing model

    # Forecast - fit
    Ntest=10
    Nhorizon = 1
    swin_params = {
        "Ntest":Ntest,
        "Nhorizon":Nhorizon,
        "Nlags":Nlags,
        "minmax" :(0,None)
        }  

    tf_params = {
        "Nepochs_i": 500,
        "Nepochs_t": 100,
        "batch_size":100
        }

    sfuvtf = sf.sforecast(y = y, model_type="tf", swin_parameters=swin_params,model=model_tf_dense, tf_parameters=tf_params)

    df_pred_uv = sfuvtf.fit(dfXYtf)

    print(f'\nmetrics = {sfuvtf.metrics}')
    dfXY_pred_uvtf = dfXYtf.join(df_pred_uv)


    y_pred = y[0]+"_pred"
    
    pred_expected = np.array([  5.57118893,  6.66552591,  8.29196644,  6.66091871,  6.85628557,
        7.43496895,  7.4591279 ,  9.36265373,  8.71832561,  8.94917297])

    pred_expected_p = pred_expected + 6
    pred_expected_m = pred_expected - 6
    pred_result = df_pred_uv[y_pred].tail(Ntest).values
    
    print()
    print(f'df_pred_uv[ypred].tail({Ntest}).values=)')
    print(df_pred_uv[y_pred].tail(Ntest).values)

    if _assert==True:
        assert (pred_result > pred_expected_m).all() , "TensorFlow Univariate forecast failed pred_result > pred_expected_m"
        assert (pred_result < pred_expected_p).all() , "TensorFlow Univariate forecast failed pred_result < pred_expected_p"
    
    return df_pred_uv


#### Multivariate M5 7 items w Exogs and Endogs 
def test_multivariate_exog_endog(_assert=True):
    

    
    dfXYtf = dfXY[covars+exogvars+le_catvars].copy()
    
    # derived attributes custom transformer
    from sklearn.base import BaseEstimator, TransformerMixin

    # pipeline/transformer development
    # BaseEstimater ... get_params(), set_params() methods
    # TransformerMixin ... fit_transform() method
    Nr = 3
    class derived_attributes_tf(BaseEstimator,TransformerMixin):
        def __init__(self, Nr = Nr): 
            self.Nr = Nr # slidig/rolling window rows
            self.dfmemory = None
            
            v1 = "unit_sales_CA1_FOODS_030"
            v2 = "unit_sales_CA1_HOUSEHOLD_416"
            v3 = "unit_sales_CA1_FOODS_393"
            self.vlist = [v1, v2, v3]
            f1 = lambda x:x+"_m1_ravg"+str(Nr)
            f2 = lambda x:x+"_m1_rstd"+str(Nr)
            self.new_attributes = [f(v) for f in (f1, f2 )for v in self.vlist]
        
        def fit(self,df):
            # ensure dataframe has enough rows
            self.dfmemory = df.tail(self.Nr) if df.index.size > self.Nr else df.index.size
                    # variables
            
            return self
        
        def transform(self,df=None, Nout=None, dfnewrows=None):
            # if df not spefified then transform on dfmemory
            # add new row(s) ... these will be provided from the predict operation
            if len(df)==0:
                df = self.dfmemory
                if isinstance(dfnewrows,pd.DataFrame):
                    df = pd.concat([df,dfnewrows])
            self.dfmemory = df.tail(self.Nr) 
            Nr=self.Nr
            dfnew=df.copy()
            
            for v in self.vlist:
                v_ravg = v+"_m1_ravg"+str(Nr)
                v_rstd =  v+"_m1_rstd"+str(Nr)
                v_m1 = v+"_m1"
                dfnew[v_ravg] = dfnew[v_m1].rolling(window=Nr).mean()
                dfnew[v_rstd] = dfnew[v_m1].rolling(window=Nr).std()

            Nclip = self.Nr
            return dfnew if Nout == None else dfnew.tail(Nout)
        
        def get_Nclip(self): # returns the number of initial rows that should be desgarded (clipped) for NaN avoidence
            return self.Nr
        
        def get_derived_attribute_names(self):
            Nr = self.Nr
            return self.new_attributes    
    
    # TensorFlow model ... Categorical Embeddings + Dense (Continuous Variables), Multiple Ouput
    Nlags = 5
    Nendogvars = 6
    Ndense = Nlags * Ncovars + Nexogvars + Nendogvars #lagged covars (does not include unlagged covars) + exogvars
    Nemb = Ncatvars
    Nembout = sum(eoutdim)
    Nout = len(covars)

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
    model_tf_dense_emb = Model(inputs=[cont_inputs, cat_inputs_list], outputs=output)

    # define optimizer and compile ...
    optimizer = Adam(learning_rate=0.05, decay=.1)
    model_tf_dense_emb.compile(loss='mse', optimizer=optimizer)

    ### fit test/strain
    y = ["unit_sales_CA1_FOODS_030", "unit_sales_CA1_HOUSEHOLD_416" ,  "unit_sales_CA1_FOODS_393" ]

    # forecast fit
    Ntest = 5
    Nhorizon = 1
    swin_params = {
        "Ntest":Ntest,
        "Nhorizon":Nhorizon,
        "Nlags":Nlags,
        "minmax" :(0,None),
        "covars":covars,
        "exogvars":exogvars,
        "catvars":le_catvars,
        "derived_attributes_transform": derived_attributes_tf
        }  

    tf_params = {
        "Nepochs_i": 500,
        "Nepochs_t": 200,
        "batch_size":100  
    }

    sfmvexen = sf.sforecast(y = y, model_type="tf", swin_parameters=swin_params,model=model_tf_dense_emb, tf_parameters=tf_params)
    
    df_pred = sfmvexen.fit(dfXYtf)
    
    pred_expected_0 =  [5.98022509,  5.47759581, 11.11475754, 12.32855797,  6.84114027 ]
    pred_expected_1 = [1.3943553 , 1.36535299, 1.47444928, 1.49734759, 1.37910318]
    pred_expected_2 = [3.9302938 , 5.07874012, 6.54496574, 5.89960051, 5.87364244]

    pred_expected_list = np.array([ pred_expected_0, pred_expected_1, pred_expected_2 ])
    
    print("y =",y)
    for n,_y in enumerate(y):
        _y_pred = _y+"_pred"
        print(_y_pred)
        
        pred_result = df_pred[_y_pred].tail(Ntest).values
        print("pred_result =", list(pred_result))

        pred_expected = pred_expected_list[n]
        print("pred_expected =",pred_expected)
        pred_expected_p = pred_expected + 4
        pred_expected_m = pred_expected - 4

        if _assert==True:
            assert (pred_result > pred_expected_m).all() , f'y = {_y_pred} TensorFlow multivariate-categorical-multi-output {n} forecast with covariates failed pred_result > pred_expected_m'
            assert (pred_result < pred_expected_p).all() , f'y = {_y_pred} TensorFlow multivariate-categorical-multi-output {n} forecast with covariates failed pred_result < pred_expected_p'
        
        
    return df_pred

def get_m7_exongroups_data():
    #### Data M5 7 items Week
    file = "../data/M5_7items_week.csv"
    sales_week_7_w = pd.read_csv(file, parse_dates = ["yearweek_dt"]).set_index("yearweek_dt")

    ### Prepare Data
    y = ["unit_sales_FOODS_3_030"]
    print(f'target variable = {y}')

    # variable types
    print("variable types ...")
    covars = [c for c in sales_week_7_w.columns if "unit_sales_" in c]
    catvars = [ "month", "event_name_1","event_name_2"]
    exogvars = [ "yeariso" , "weekiso",  "snap_CA",  "CCI_USA", "sell_price_FOODS_3_030"]
    exogs1 = [ "yeariso" , "weekiso",  "snap_CA",  "CCI_USA" ]
    exogs2 = ["sell_price_FOODS_3_030"]
    Ncatvars = len(catvars)
    Ncovars = len(covars)
    Nexogvars = len(exogvars)
    exenvars = [exogs1, exogs2]

    # dfXY ... covars + exogvars + catvars
    cols = covars+catvars+exogvars
    colsdp = cols  #+ ["yearweek_dt"]
    dfXYdp = sales_week_7_w[colsdp].copy()
    dfXY = sales_week_7_w[cols].copy()

    # label Encoding
    le_catvars = [ "le_"+ c for c in catvars ] # label encoded category columns
    le = LabelEncoder()
    dfXY[le_catvars] = dfXY[catvars].apply(le.fit_transform)

    # embedding dimensions
    eindim = [dfXY[le_catvars].groupby(c)[c].count().index.size + 1 for c in le_catvars] # add 1 to the dim or err in TF
    eoutdim = [np.rint(np.log2(x)).astype(int) for x in eindim]

    dfXY = dfXY[covars+exogvars+le_catvars]
    
    return dfXY, covars, exogvars, exenvars, le_catvars, exogs1, exogs2, Ncovars, Ncatvars, Nexogvars, eoutdim, eindim

def test_exengroups(_assert=True):
    
    dfXY, covars, exogvars, exenvars, le_catvars, exogs1, exogs2, Ncovars, Ncatvars, Nexogvars, eoutdim, eindim  = get_m7_exongroups_data()
    
    y = ["unit_sales_FOODS_3_030"]
    print(f'target variable = {y}')
    # Fit
    Ntest = 3
    Nlags=5
    swin_params = {
        "Ntest":Ntest,
        "Nhorizon":1,
        "Nlags":Nlags,
        "minmax" :(0,None),
        "covars":covars,
        "catvars":le_catvars,
        "exogvars":exogvars,
        "exenvars":exenvars
        }  

    tf_params = {
        "Nepochs_i": 500,
        "Nepochs_t": 200,
        "batch_size":100  
    }
    
    ### TensorFlow Model
    Nlags = 5
    Ndense1 = Nlags * Ncovars 
    Ndense2 = len(exogs1)
    Nlinear = len(exogs2) 
    Nemb = Ncatvars
    Nembout = sum(eoutdim)
    Nout = Ncovars

    #Ndense = Nlags  # N continous/dense variables, in this case covars is 1 (univarate)

    # Dense Network, 2 hidden layers, continuous variables ... covar lags and exogenous variables
    covarlags_in = Input((Ndense1,))
    hcovarlags = Dense(Ndense1, activation='relu')(covarlags_in)

    exogs1_in = Input((Ndense2,))
    hexogs1 = Dense(Ndense2, activation='relu')(exogs1_in)

    exogs2_in = Input((Nlinear,))
    hexogs2 = Dense(Nlinear)(exogs2_in)


    # embeddings, cat vars
    cat_inputs_list = [ Input((1,)) for c in range(Nemb) ]  # one embedding for each categorical variable
    emb_out_list = [Embedding(ein,eout,input_length=1)(cat) for ein,eout,cat in zip(eindim ,eoutdim,cat_inputs_list) ]
    emb_flat_list = [Flatten()(emb_out) for emb_out in emb_out_list ]

    # combined 
    combined = concatenate([hcovarlags]+emb_flat_list + [hexogs1] )
    combined_d = Dropout(0.2)(combined)

    # dense reduction layers
    Nh1c = Ndense1 + Nembout + Ndense2 # 
    h1c = Dense(Nh1c, activation='relu')(combined_d)
    h1c_d = Dropout(0.2)(h1c)
    Nh2c = np.rint(Nh1c/2).astype(int)
    h2c = Dense(Nh2c, activation='relu')(h1c_d)
    h2c_d = Dropout(0.2)(h2c)

    # combined to output ... combine the hidden reduced variables and the linear
    combined_to_out = concatenate([h2c_d]+[hexogs2])

    # output
    output = Dense(Nout)(combined_to_out)  # linear activation ... linear combination 
    model_tf_dense2_emb_so = Model(inputs=[covarlags_in, exogs1_in, exogs2_in, cat_inputs_list], outputs=output)

    # define optimizer and compile
    optimizer = Adam(learning_rate=0.07, decay=.2)
    model_tf_dense2_emb_so.compile(loss='mse', optimizer=optimizer)
    

    sf_tf_dense_emb_so = sf.sforecast(y = y, model_type="tf", swin_parameters=swin_params,model=model_tf_dense2_emb_so, tf_parameters=tf_params)

    df_pred = sf_tf_dense_emb_so.fit(dfXY)
    
    pred_expected =  np.array([50.5, 46.3, 34.5 ])
    
    print()
    print(f'print_expected = {pred_expected}')
    
    y_pred = y[0]+"_pred"

    pred_expected_p = pred_expected + 10
    pred_expected_m = pred_expected - 10
    pred_result = df_pred[y_pred].tail(Ntest).values
    
    print()
    print(f'print_result = {pred_result}')

    if _assert == True:
        assert (pred_result > pred_expected_m).all() , "TensorFlow Univariate forecast failed pred_result > pred_expected_m"
        assert (pred_result < pred_expected_p).all() , "TensorFlow Univariate forecast failed pred_result < pred_expected_p"

    return df_pred