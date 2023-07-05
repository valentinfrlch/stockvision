# we're going to use temporal fusion transformers to predict the future of multiple time series.
# The time series are stocks with daily returns


# libraries:
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline


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
    
    max_prediction_length = 24
    max_encoder_length = 7*24
    print("********")
    print(dataset["time_idx"])
    print("********")
    training_cutoff = dataset["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        dataset[lambda x: x.hours_from_start <= training_cutoff],
        time_idx="time_idx",
        target="rtn",
        group_ids=["uid"],
        min_encoder_length=max_encoder_length // 2, 
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=['rtn'],
        target_normalizer=GroupNormalizer(
            groups=["uid"], transformation="softplus"
        ),  # we normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, dataset, predict=True, stop_randomization=True)
    
    # create dataloaders for  our model
    batch_size = 64 
    # if you have a strong GPU, feel free to increase the number of workers  
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    validation_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=2)


    actuals = torch.cat([y for x, (y, weight) in iter(validation_dataloader)]).to("cuda")
    baseline_predictions = Baseline().predict(validation_dataloader)
    (actuals - baseline_predictions).abs().mean().item()
    
    
    
    
def predict(lookback, forward):
    pass


if __name__ == "__main__":
    data = preprocess()
    train(data)