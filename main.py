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
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline, TemporalFusionTransformer, QuantileLoss, MAE


def preprocess():
    # load the data from .csv file
    path = "dataset/returns.csv"
    # the limiter is a semicolon
    df = pd.read_csv(path, sep=";")

    # filter out all UIDs that are not in whitelist
    whitelist = ["B018KB-R_1", "B01HWF-R_2", "B029D5-R_1",
                 "B3GKSL-R_1", "B3K1X9-R_1", "B3LCDJ-R_2"]
    df = df[df["UID"].isin(whitelist)]

    # align all the stocks by date
    df_time = pd.DataFrame({"Date": df.Date.unique()})
    df_time.sort_values(by="Date", inplace=True)
    df_time.reset_index(drop=True, inplace=True)
    df_time["idx"] = list(df_time.index)
    df = pd.merge(df, df_time, on=["Date"], how="inner")

    data = pd.DataFrame(
        dict(
            # convert Date to datetime
            date=lambda_date(df["Date"]),
            time_idx=df.index,
            uid=df["UID"],
            rtn=df["Return"],
        )
    )

    return data


def lambda_date(dates):
    # dates is a list of 8 digit numbers
    # Format is YYYYMMDD -> datetime object
    for i in range(len(dates)):
        dates[i] = pd.to_datetime(str(dates[i]), format="%Y%m%d")
    return dates


def visualize(data):
    # use matplotlib to visualize the data, different stocks have different colors

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


def forecast(data, lookback=30, horizon=30):
    # TRAINING
    max_prediction_length = 30
    max_encoder_length = 90
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
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True)

    # create dataloaders
    batch_size = 64
    training_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=6)
    validation_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=6)

    torch.set_float32_matmul_precision('medium')  # todo: set to 'high'
    actuals = torch.cat(
        [y for x, (y, weight) in iter(validation_dataloader)]).to("cuda")
    baseline_predictions = Baseline().predict(validation_dataloader)
    (actuals - baseline_predictions).abs().mean().item()

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=5,  # todo: MAX EPOCHS
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
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4)

    trainer.fit(
        tft,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print("Best model path:", best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # PREDICTION
    # evaluate on training data

    predictions = best_tft.predict(
        validation_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
    MAE()(predictions.output, predictions.y)

    raw_predictions = best_tft.predict(
        validation_dataloader, mode="raw", return_x=True)
    print("Raw Prediction Fields:")
    print(raw_predictions._fields)
    print('\n')
    print(raw_predictions.output.prediction.shape)

    # list of all the unique uids
    uids = data["uid"].unique()
    
    for uid in uids:
        fig, ax = plt.subplots(figsize=(10, 5))

        raw_prediction= best_tft.predict(
            training.filter(lambda x: (x.uid == uid)),
            mode="raw",
            return_x=True,
        )
        best_tft.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0, ax=ax);
        # set the title to the uid
        ax.set_title(uid)
        # save the figure to /results
        fig.savefig(f"results/{uid}.png")
        


if __name__ == "__main__":
    data = preprocess()
    # visualize(data)
    forecast(data)
