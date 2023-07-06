# we're going to use temporal fusion transformers to predict the future of multiple time series.
# The time series are stocks with daily returns


# libraries:
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline, TemporalFusionTransformer, QuantileLoss


def preprocess():
    # load the data from .csv file
    path = "dataset/returns.csv"
    # the limiter is a semicolon
    df = pd.read_csv(path, sep=";")
    
    # align all the stocks by date
    df_time = pd.DataFrame({"Date": df.Date.unique()})
    df_time.sort_values(by="Date", inplace=True)
    df_time.reset_index(drop=True, inplace=True)
    df_time["idx"] = list(df_time.index)
    df = pd.merge(df, df_time, on=["Date"], how="inner")
    
    print(df.head(10))
    
    
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

def visualize(data):
    # use matplotlib to visualize the data as a line graph
    # x-axis is the date, y-axis is the return
    # use different colors for different stocks (UIDs)
    plt.figure(figsize=(15, 8))
    for uid in data["uid"].unique():
        plt.plot(
            data["date"][data["uid"] == uid],
            data["rtn"][data["uid"] == uid],
            label=uid,
        )
    plt.title("Returns")
    plt.legend()
    plt.show()



def train(data):    
    max_prediction_length = 24
    max_encoder_length = 7*24
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
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
            groups=["uid"], transformation="count"
        ),  # we normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    
    # create dataloaders
    batch_size = 64  
    training_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=12)
    validation_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=12)


    torch.set_float32_matmul_precision('medium') # todo: set to 'high'
    actuals = torch.cat([y for x, (y, weight) in iter(validation_dataloader)]).to("cuda")
    baseline_predictions = Baseline().predict(validation_dataloader)
    (actuals - baseline_predictions).abs().mean().item()
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()  
    logger = TensorBoardLogger("lightning_logs")  

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu', 
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=160,
        output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10, 
        reduce_on_plateau_patience=4)

    trainer.fit(
        tft,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader)
    
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
     
    
    
    
def predict(lookback, forward):
    pass


if __name__ == "__main__":
    data = preprocess()
    visualize(data)
    # train(data)
    