# we're going to use temporal fusion transformers to predict the future of multiple time series.
# The time series are stocks with daily returns


# libraries:
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def preprocess():
    # load the data from .csv file
    path = "dataset/returns.csv"
    # the limiter is a semicolon
    df = pd.read_csv(path, sep=";")
    
    data = pd.DataFrame(
        dict(
            # convert Date to datetime
            date = lambda_date(df["Date"]),
            time_idx = df.index,
            uid = df["UID"],
            rtn = df["Return"],
        )
    )
    
    return data


def lambda_date(dates):
    # dates is a list of 8 digit numbers
    # first 4 digits are the year, next 2 are the month, last 2 are the day
    for i in range(len(dates)):
        dates[i] = pd.to_datetime(str(dates[i]), format="%Y%m%d")
    return dates



def train(data):
    # create dataset:
    dataset = TimeSeriesDataSet(
        data,
        group_ids=["uid"],
        target="rtn",
        time_idx="time_idx",
        max_encoder_length=2,
        max_prediction_length=3,
        time_varying_unknown_reals=["rtn"],
        allow_missing_timesteps=True,
        target_normalizer=None
    )

    # pass to dataloader
    dataloader = dataset.to_dataloader(batch_size=1)

    #load the first batch
    x, y = next(iter(dataloader))
    
    
    
    
def predict(lookback, forward):
    pass


if __name__ == "__main__":
    data = preprocess()
    train(data)