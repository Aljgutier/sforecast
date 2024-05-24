import pandas as pd
import sforecast as sf

#### data Superstore
df_sales = pd.read_csv("../data/Superstore_subcatsales_2017_cdp.csv", parse_dates = ["Order Date"])
aggs = {
    "Sales":"sum",
    "Quantity":"sum"  
}

print(df_sales)