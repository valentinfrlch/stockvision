# we're going to use temporal fusion transformers to predict the future of multiple time series.
# The time series are stocks with daily returns


# libraries:
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def preprocess():
    # load the data from .csv file
    path = "dataset/returns.csv"
    df = pd.read_csv(path)
    print(df.iloc[:,0])
    """
    data = pd.DataFrame(
        dict(
            time_idx = df["Date"],
            uid = df["UID"],
            rtn = df["Return"],
        )
    )
    
    return data
"""
preprocess()



def train(data):
    # create dataset:
    dataset = TimeSeriesDataSet(
        data,
        group_ids=["group"],
        target="target",
        time_idx="time_idx",
        max_encoder_length=2,
        max_prediction_length=3,
        time_varying_unknown_reals=["target"],
        static_categoricals=["holidays"],
        target_normalizer=None
    )

    # pass to dataloader
    dataloader = dataset.to_dataloader(batch_size=1)

    #load the first batch
    x, y = next(iter(dataloader))

