# sdafasf
from typing import Callable, Any, Union
import pandas as pd
from dataclasses import dataclass
import numpy as np


# derived attributes custom transformer
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass(init=True)
class rolling_transformer(BaseEstimator,TransformerMixin):

    variable_transform_dict: dict[str:(Callable)]
    Nrw: int = 3
    dfmemory: pd.DataFrame = pd.DataFrame()

    def __post_init__(self):
        self.dfmemory = pd.DataFrame()
        
        # dictionaries
        # -- derivedvar to variable 
        # -- derivedvar to funcstr
        self.derivedvar_var_dict = {}
        self.derivedvar_funcstr_dict = {}
        for item in self.variable_transform_dict.items():
            v = item[0]
            tfr = item[1]
            if isinstance(tfr,list):
                for t in tfr:
                    func_str = t
                    derived_variable = v+"_m1_"+func_str+str(self.Nrw)
                    self.derivedvar_var_dict[derived_variable]= v
                    self.derivedvar_funcstr_dict[derived_variable] = func_str
            else:
                func_str = tfr
                derived_variable = v+"_m1_"+func_str+str(self.Nrw)
                self.derivedvar_var_dict[derived_variable] = v
                self.derivedvar_funcstr_dict[derived_variable]= func_str
           
        # derived_variables_list
        self.derived_variables_list = [dv for dv in self.derivedvar_var_dict.keys()]
        
             
    def fit(self, df):
        
        self.dfmemory = df.tail(self.Nrw) if df.index.size > self.Nrw else df
        
        return self
        
    def transform(self, 
            df:pd.DataFrame, 
            Nout: Union[int,None] = None,
            dfnewrows: pd.DataFrame = pd.DataFrame())  -> pd.DataFrame:
        
        # add new rows to df
        if len(df) == 0:
            df = self.dfmemory
            if len(dfnewrows) > 0:
                df = pd.concat([df, dfnewrows])
        
        self.dfmemory = df.tail(self.Nrw) # save to transformer state
        
        Nrw = self.Nrw # get rolling N
        dfnew = df.copy() # copy df ... add new variables
        
        for item in self.derivedvar_var_dict.items():
            v_derived = item[0]
            v = item[1]
            func_str = self.derivedvar_funcstr_dict[v_derived]
            v_m1 = v+"_m1"  # name of input shift variable ("minus 1")
            
            if func_str == "mean":
                dfnew[v_derived] = dfnew[v_m1].rolling(window=Nrw).mean()
            elif func_str == "std":
                dfnew[v_derived] = dfnew[v_m1].rolling(window=Nrw).std()
            else:
                print('Warning: derived variables = NaN')
                dfnew[v_derived] = np.NaN
           
        df_return= dfnew if Nout is None else dfnew.tail(Nout)
          
        #print(f'df_return, Nout ={Nout}' )
        #display(df_return)          
                            
        return df_return
                     
    def get_Nclip(self):
        return self.Nrw
    
    def get_derived_attribute_names(self):
        return self.derived_variables_list
