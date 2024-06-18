import pandas as pd
from dataclasses import dataclass
from datetime import datetime

# Method Chaining
# https://stackoverflow.com/questions/41817578/basic-method-chaining

@dataclass(init=True)
class transforms():
    
    # init variabbles
    _df: pd.DataFrame
    datetime_column: str = ""
    inplace: bool = False
    
    # init procedures
    def __post_init__(self):
        if self.inplace == True:
            self.df = self._df
        else:
            self.df = self._df.copy()
            
        self.period_col = "p"

    # aggregate to period
    def frequency(self,
                datetime_column:str, 
                aggs: dict[str,str],
                frequency: str, 
                group_by: list[str],
                tp_column: str = "tp"
                ):

        """Receives a DataFrame time-series (with a datetime colomn), The output includes the "time period" (prefix_tp) datetime column  with the output attributes (columns) grouped as specified by the aggs input.

        Args:
            df (pd.DataFrame): timeseires Dataframe with a datetime column and  attribute columns to be aggregated to the indicated time frequency.
            datetime_column (str): Datetme column to be used for deriving the new time period according to the specified frequency.
            frequency (str): frequency = "D", "W", "M", defaults to "D".
            tp_column (str): naem of the time period column. The time period is derived from the input datetime_column with pandas.to_period(). The time period
             is set to the beginning of the period corresponding to the specified frequency.
            aggs (dict): dictionary defining the column group by aggregation, e.g., {"col1":"sum", "col2":"sum"}
            group_by (list: str): list of columns to add to the group by, in addition to the period column.

        Returns:
            pd.DataFrame: DataFrame aggregated to the desired period with output column "p" (by default) and attribute columns corresponding to the aggs()input dictionary.
        """
        
        assert frequency == "D" or frequency== "M" or frequency == "W" , f'frequency = {frequency} not valid. The supported values of frequency are "D" or "W", or "M'
        
    
        # below is Monday start ... reference is for Sunday Week start
        # https://stackoverflow.com/questions/42586916/how-to-use-python-to-convert-date-to-weekly-level-and-use-sunday-as-week-start-d 

        # df_period 
        df_period = self.df
        self.datetime_column = tp_column
    
        # Period Column 
        #  - "W" gets the beginning and end of period
        df_period[tp_column] = df_period[datetime_column].dt.to_period(frequency)
        
        
        # get the beginnig of week from period
        if frequency =="W":
            df_period[tp_column] = pd.to_datetime([i[0] for i in df_period[tp_column].astype(str).str.split("/").values])
        else:
            df_period[tp_column] = df_period[tp_column].dt.to_timestamp()
            
        # Group By
        groups = [tp_column] + group_by
        df_period = df_period.groupby(groups).agg(aggs).reset_index()
        
        self.df = df_period
        
        return self

    # filter in and/or filter out
    def filter(self,
            filter_in: dict[str,list] = [], 
            filter_out: dict[str,list] = []
            ):
        """filter in or filter out from the dataframe.

        Args:
            filter_in (dict[str,list], optional): Key, attribute (i.e., column), value (list) filter specification. If column value is included in the list, then the corresponding row is kept in the output. Defaults to [], nothing filtered.
            
            filter_out (dict[str,list], optional): Key attribute (i.e., column) to be filtered by value (list). If the attribute (column) value is in the list then the corresponding row is removed from the output. Defaults to [], nothing filtered.

        Returns:
            pd.DataFrame: filtered dataframe
        """
        
        # df_dfiltered 
        df_filtered = self.df
        
        if len(filter_in) > 0:
            for k in filter_in.keys():
                df_filtered = df_filtered[df_filtered[k].isin(filter_in[k])]
        
        if len(filter_out) > 0:
            for k in filter_out.keys():
                df_filtered = df_filtered[~df_filtered[k].isin(filter_out[k])]
                
        self.df = df_filtered

        return self
    
    # reset to original dataframe
    def filter_datetime(self, datetime_start:str="",datetime_end:str=""):
        """
           The `data_transforms.filter_datetime()` is useful when preparing data for time-series forecasting and/or a datetime column.
        
           filter rows to include from a gieven period start
           or to include periods upt to given period end.
           
           The datetime_column must either be specified during class instantiation or by calling`data_transforms.frequency()`.
           
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
    
    #TODO
    def filter_rows(self):
        
        return self
    
    def return_df(self):
        return self.df