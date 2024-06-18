# ML Data Prep

from typing import Any
import pandas as pd
from dataclasses import dataclass
import numpy as np
import regex as re
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# create time weries
#'forecast_date':pd.date_range(begin_date, periods=len(units))})
def get_timeseries(date_column:str, begin_date:str, periods:int, freq:str="D") -> pd.DataFrame:
    """create a pandas time series using pandas.date_range()

    Args:
        date_column (str): date column of the time-series
        begin_date (str): date to start the time-series
        periods (int): number of periods
        freq (str, optional): time-series frequency. Defaults to "D".

    Returns:
        pd.DataFrame: _description_
    """

    _freq = freq if freq != "M" else "MS"
    
    print(begin_date)
    df = pd.DataFrame({date_column: pd.date_range(begin_date, periods=periods, freq=_freq)})
    
    return df

#vv------------- datepart -------------vv#
# inspired and modified from Fastai
# from Fastai
# modified for time series data
#    the attr depending on the ts_period setting: D, W, M
# https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
        
    return None
        
def get_is_start(df:pd.DataFrame, col_is_start:str) -> None:
    
    # temp column
    df["prev_"+col_is_start] = df[col_is_start].shift(1)
    
    # is start
    is_start = np.where( df[col_is_start] != df["prev_"+col_is_start], 1.0 , 0)
    is_start[0] = np.NaN

    df.drop("prev_"+col_is_start, inplace=True, axis=1)
    
    return is_start


def get_is_end(df:pd.DataFrame, col_is_end:str) -> None:
    
    # temp column
    df["next_"+col_is_end] = df[col_is_end].shift(-1)
    
    # is start
    is_end = np.where( df[col_is_end] != df["next_"+col_is_end], 1.0, 0)
    
    is_end[-1] = np.NaN
    # drop temp column
    df.drop("next_"+col_is_end, inplace=True, axis=1)
    
    return is_end
    
        
def add_datepart(df, field_name,
                 prefix=None, 
                 drop=True, 
                 time=False, 
                 period: str = 'D') -> pd.DataFrame:
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    
    
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    
    # attr list based on ts_period
    # defaults to the fastai in the else branch
    if period != "D" and period=="W":
            attr = ['Year', 'Quarter',  'Month', 'Week']
            
    elif period != "D" and  period == "M":
            attr = ['Year', 'Quarter', 'Month']
    else:
        attr = ['Year','Quarter' , 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
        'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    

    
    # Make the Date Parts
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: 
        df[prefix + n] = getattr(field.dt, n.lower()).astype(int) if n != 'Week' else week
    mask = ~field.isna()

    
    if period != "D" and period == "W":

        # Month start/end
        df[prefix +"Is_month_start" ] = get_is_start(df, prefix+"Month")
        df[prefix +"Is_month_end" ] = get_is_end(df,prefix+"Month")
        # Quarter start/end
        df[prefix +"Is_quarter_start" ] = get_is_start(df, prefix+"Quarter")
        df[prefix +"Is_quarter_end" ] = get_is_end(df, prefix+"Quarter")
        # Year start/end
        df[prefix +"Is_year_start" ] = get_is_start(df, prefix+"Year")
        df[prefix +"Is_year_end" ] = get_is_end(df, prefix+"Year")


    elif period != "D" and period == "M":
        # Month starts at 1
        
        df[prefix +"Is_Quarter_start"] = np.where( df[prefix+"Month"].isin([1, 4, 7, 10]), 1 , 0)
        
        df[prefix +"Is_Quarter_end"] = np.where( df[prefix+"Month"].isin([3, 6, 0, 12]), 1 , 0)
        
        df[prefix + "Is_year_start"] = get_is_start(df, prefix + "Year")
        df[prefix + "Is_year_end"] = get_is_end(df,prefix+"Year")

    # elapsed 
    #    only if ts_period == D ... defaults to fastai non time-series
    else:
        df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,np.nan)
        
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

#^^------------- datepart -------------^^#

# Save Enccders ... for Future reference
# https://stackoverflow.com/questions/56251731/how-to-save-one-hot-encoded-model-and-predict-new-unencoded-data-using-scikitlea


#vv------------- wide format -------------vv#
def to_flat_columns(hier_cols: str) -> pd.DataFrame:
    """flatten hierarchical columns

    Args:
        hier_cols (str): hierarchical columns, for example typically resulting from a pivot operation. Each hiearchical column name is a combination of the hierical_colum + _ + sub_category

    Returns:
        pd.DataFrame: dataframe with flat columns
    """
    flat_cols=[]
    for clist in hier_cols:
        for n,ci in enumerate(clist):
            c = ci if n == 0 else c+"_"+ci 
        flat_cols.append(c)
    return flat_cols


def get_df_wide(_df, index_col:str,  
                category_column: str,  
                value_columns: list[str])->pd.DataFrame:
    
    """Generate a "wide" format dataframe. Pivot the input dataframe based on a categorical column with a column for each categorical value.


    Args:
        index_col: dataframe index column
        category_column: categorical pivot column 
        value_columns: value columns corresponding to the pivot columns

    Returns:
        DataFrame: wide format dataframe
    """
    
    dfw = _df
    
    dfw = dfw.pivot(index=index_col, columns = category_column, values = value_columns )
    
    dfw.columns = to_flat_columns(dfw.columns)
    
    dfw = dfw.fillna(0).reset_index()

    return dfw
#^^------------- wide format -------------^^#

@dataclass(init=True)
class transforms():
    _df: pd.DataFrame
    datetime_column: str = ""
    frequency: str = "D"
    inplace: bool = False
    
    
    def __post_init__(self):
        
        if self.inplace == True:
            self.df = self._df
        else:
            self.df = self._df.copy()
        
    def fill_na(self):
        return self.df
        
    def train_test_split(self):
        return self.df
    
    def ml_columns(self, columns_keep: [str] =[], columns_remove: [str] = []) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            columns_keep (str], optional): keep the columns in the specified list. columns_keep takes precedence over columns_remove. Defaults to [].
            columns_remove (str], optional): drop the specified columns. Defaults to [].

        Returns:
            pd.DataFrame: _description_
        """
        
        df = self.df
        
        if len(columns_keep) > 0:
             df = df[columns_keep].copy()
        elif len(columns_remove) > 0:
            df.drop(columns_remove, axis=1, inplace=True)
        
        self.df = df
        
        return self
    
    # based on fastai's add_datepart
    def datepart(self, drop: bool = False)->pd.DataFrame:
        
        df = self.df
        datetime_column = self.datetime_column
        frequency = self.frequency
        
        prefix = datetime_column + "_"
        
        df = add_datepart(df,datetime_column, prefix=prefix,drop=drop, period=frequency)
        
        self.df = df
        
        return self
    
    def wide_format(self, 
                index_col:str,  
                category_column: str,  
                value_columns: list[str]):
        
        """Generate a "wide" format dataframe. Pivot the input dataframe based on a categorical column. Generates a value column for each categorical value.


        Args:
            index_col: dataframe index column
            category_column: categorical pivot column 
            value_columns: value columns corresponding to the pivot columns
        """
        
        df = self.df
        df = get_df_wide(df, index_col, category_column, value_columns ) 
        self.df = df
        
        return self
    
    def label_encoder(self,cat_columns: list[str])->pd.DataFrame:
        """Label encode categorical variable columns
        Args:
            cat_columns (list[str]): List of categorical columns

        Returns:
            pd.DataFrame: dataframe with corresponding label encoded categorical columns
        """
        
        encoder = LabelEncoder()
        
        df = self.df
        # label encoder does 1 column at a time
        for col in cat_columns:
            df[col] = encoder.fit_transform(df[col])
            
        self.df = df
        
        return self
    
    def onehot_encoder(self, cat_columns: list[str], drop: bool=True) -> pd.DataFrame:
        """one hot encode categorical columns. Optinally drop  the original categorical column (not one hot). The returned one hot encoded column names
        are prefixed with the origin categorical column name ("columname_").

        Args:
            cat_columns (list[str]): _description_
            drop (bool, optional): optionally drop the categorical columns. Defaults to True.

        Returns:
            pd.DataFrame: one hot encoded dataframe. Each one-hot encoded colum is pre-fixed with the original categorical column name "cat_column_"
        """
        df = self.df
        
        encoder = OneHotEncoder(handle_unknown='ignore')
    
        indx = df.index
        
        for col in cat_columns:
            _df = pd.DataFrame(encoder.fit_transform(df[[col]]).toarray(), index=indx)
            one_hot_cols = _df.columns.tolist()
            _df.columns = [col + "_" + str(one_hot) for one_hot in one_hot_cols]
            df = df.join(_df)
            
        if drop:
            df = df.drop(cat_columns, axis=1)
        
        self.df = df
        return self
    
     # reset to original dataframe
    def filter_datetime(self, datetime_start:str="",datetime_end:str=""):
        """
           The `mlprep.filter_datetime()` method is useful when preparing data for time-series forecasting and/or a datetime column.
        
           filter rows to include from a gieven period start
           or to include periods up to given period end.
           
           the datetime_column is specified in the mlprep.transforms() class constructur. IF the datetime_column is not specified then there is effectively no filtering.
           
        Args:
            datetime_start (str): start of period inclusive, defaults to "" (no filtering)
            
            datetime_end (str): end of period before (non-inclusive), defaults to "" (no filtering)

        Returns:
            pd.DataFrame filtered to the indicated date range
        """
    
        df = self.df
        datetime_column = self.datetime_column

        if datetime_start:
            start = pd.to_datetime(datetime_start)
            df = df[df[datetime_column] >= start]
                                  
        if datetime_end:
            end = pd.to_datetime(datetime_end)
            df = df[df[datetime_column] < end]
            
        self.df = df
        
        return self
    
    def set_index(self, index_column:str):
        
        df = self.df
        
        df = df.set_index(index_column)
        
        self.df = df
        
        return self
    
    #TODO
    def filter_rows(self):     
        return self
        
    #TODO
    def scale(self):
        return self
    
    def return_df(self) -> pd.DataFrame:
        return self.df

    