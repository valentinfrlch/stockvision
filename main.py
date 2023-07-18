
# libraries:
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline, TemporalFusionTransformer, QuantileLoss, MAE, NaNLabelEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess():
    # load the data from .csv file
    path = "dataset/returns_200.csv"
    # the limiter is a semicolon
    df = pd.read_csv(path, sep=";")
    df["weekday"] = pd.to_datetime(df["date8"], format="%Y%m%d").dt.weekday
    df["month"] = pd.to_datetime(df["date8"], format="%Y%m%d").dt.month

    # filter out all UIDs that are not in whitelist
    # whitelist = ["B018KB-R_1", "B01HWF-R_2", "B029D5-R_1", "B0TXKG-R_1", "B16HJ6-R_1", "B18RVB-R_1"]
    # df = df[df["uid"].isin(whitelist)]
    

    # align all the stocks by date
    df_time = pd.DataFrame({"date8": df.date8.unique()})
    df_time.sort_values(by="date8", inplace=True)
    df_time.reset_index(drop=True, inplace=True)
    df_time["idx"] = list(df_time.index)
    df = pd.merge(df, df_time, on=["date8"], how="inner")
    
    # make sure numbers are interpreted as numbers
    df["tr"] = pd.to_numeric(df["tr"], errors="coerce")
    df["date8"] = pd.to_numeric(df["date8"], errors="coerce")
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    
    # todo: add features like weekday, month etc and add them to static_categoricals

    data = pd.DataFrame(
        dict(
            # convert Date to datetime
            date=df["date8"],
            time_idx=df["idx"],
            weekday=df["weekday"],
            month = df["month"],
            uid=df["uid"],
            tr=df["tr"],
        )
    )
    
    print(df.head(10))

    return data


def lambda_date(dates):
    # dates is a list of 8 digit numbers
    # Format is YYYYMMDD -> datetime object
    for i in range(len(dates)):
        dates[i] = pd.to_datetime(str(dates[i]), format="%Y%m%d")
    return dates
    

def visualize(data):
    # use matplotlib to visualize the data, different stocks have different colors

    for uid in data["uid"].unique():
        plt.figure(figsize=(15, 8))
        # x axis is time_idx, y axis is rtn
        plt.plot(
            data[data["uid"] == uid]["time_idx"],
            data[data["uid"] == uid]["tr"],
            label=uid,
        )
        # break # plot only the first chart
        plt.title("Returns {uid}")
        # save the figure to /results
        plt.savefig(f"results/visualize_{uid}.png")


def forecast(data, max_encoder_length=365, max_prediction_length=30):
    # TRAINING
    training_cutoff = data["time_idx"].max() - max_prediction_length
    print("Training cutoff:", training_cutoff, "\n")

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="tr",
        group_ids=["uid"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["uid"],
        time_varying_known_reals=["time_idx", "date", "month", "weekday"],
        time_varying_unknown_reals=['tr'],
        target_normalizer=GroupNormalizer(
            groups=["uid"], transformation="count"
        ),  # we normalize by group
        categorical_encoders={
            "uid": NaNLabelEncoder(add_nan=True)  # special encoder for categorical target
        },
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
        train=True, batch_size=batch_size, num_workers=12)
    validation_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=12)
    
    
    #debug
    """
    x, y = next(iter(training_dataloader))
    print(x['encoder_target'])
    print(x['groups'])
    print('\n')
    print(x['decoder_target'])
    """

    torch.set_float32_matmul_precision('medium')  # todo: set to 'high'
    actuals = torch.cat(
        [y for x, (y, weight) in iter(validation_dataloader)]).to(device=device)
    baseline_predictions = Baseline().predict(validation_dataloader)
    (actuals - baseline_predictions).abs().mean().item()

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=2, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator=device,
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

    # predictions = best_tft.predict(
    #     validation_dataloader, return_y=True, trainer_kwargs=dict(accelerator=device))
    # MAE()(predictions.output, predictions.y)

    # raw_predictions = best_tft.predict(
    #     validation_dataloader, mode="raw", return_x=True)
    
    
    predictions = best_tft.predict(training_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    
    """
    DEBUGGING
    print("Raw Prediction Fields:")
    print(raw_predictions._fields)
    print('\n')
    print(raw_predictions.output.prediction.shape)
    """

    # list of all the unique uids
    uids = data["uid"].unique()

    for uid in uids:
        fig, ax = plt.subplots(figsize=(10, 5))

        raw_prediction = best_tft.predict(
            training.filter(lambda x: (x.uid == uid)),
            mode="raw",
            return_x=True,
        )
        best_tft.plot_prediction(
            raw_prediction.x, raw_prediction.output, idx=0, ax=ax)
        # set the title to the uid
        ax.set_title(uid)
        # save the figure to /results
        fig.savefig(f"results/prediction_{uid}.png")


if __name__ == "__main__":
    data = preprocess()
    # visualize(data)
    forecast(data)
