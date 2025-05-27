import pandas as pd
import os

# Loading reference relationships
cit_df = pd.read_csv(
    "data/cit-HepTh.txt",
    sep="\t",
    header=None,
    names=["source", "target"]
)

# Loading submission time
dates_df = pd.read_csv(
    "data/cit-HepTh-dates.txt",
    sep="\t",
    header=None,
    names=["paper_id", "date"]
)
dates_df["year"] = pd.to_datetime(dates_df["date"]).dt.year