import pandas as pd
from main import forecast

def preprocess():
    # load the dateset 
    path = "dataset/returns_200.csv"
    df = pd.read_csv(path, sep=";")
    
    # we want to stack all the different uids after each other
    # we have an "idx" column that we can use to sort the data
    # the index starts at 0 for each uid, so we can use that to sort
    
    df_time = pd.DataFrame({"date8": df.date8.unique()})
    df_time.sort_values(by="date8", inplace=True)
    df_time.reset_index(drop=True, inplace=True)
    df_time["idx"] = list(df_time.index)
    df = pd.merge(df, df_time, on=["date8"], how="inner")

    print(df.head(10))

    
    



if __name__ == "__main__":
    preprocess()
    # forecast(data)