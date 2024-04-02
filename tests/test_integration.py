import pandas as pd
import numpy as np

np.random.seed(0)

start_date = "2020-01-01"
n_days = 120
n_train = 100
n_test = n_days - n_train

date_range = pd.date_range(start_date, periods=n_days)
target = pd.Series(np.random.normal(size=n_days) / 100 + np.sin(np.linspace(0, 3, n_days)), index=date_range)

past_known_covariates = pd.DataFrame(
    {
        "past_1": np.random.normal(size=n_train),
        "past_2": target.iloc[:n_train] - np.random.normal(0.1, 0.1, size=n_train),
    },
    index=date_range[:n_train],
)

future_known_covariates = pd.DataFrame(
    {
        "future_1": np.random.normal(size=n_days),
        "future_2": target - np.random.normal(0.4, 0.01, size=n_days),
    },
    index=date_range,
)
