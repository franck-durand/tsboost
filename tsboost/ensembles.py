import pandas as pd
import numpy as np
from tsboost import TSRegressor

#h

class Stacker(object):

    def __init__(self,
                 tsregressors,
                 *args,
                 **kwargs):


        self.tsregressors = self.check_regressors(tsregressors)
        self.forecast_name = "forecast"

    def check_regressors(self, regressors):

        if type(regressors) != list:
            raise Exception('input must be list')

        for regressor in regressors:
            if not isinstance(regressor, TSRegressor):
                raise Exception('list objects must be instances of tsboost.TSRegressor')

        return regressors

    def fit_predict(self, data, cv_dates=[None], *args, **kwargs):

        data_finale = pd.DataFrame()
        for i, tsregressor in enumerate(self.tsregressors):
            data_forecast = tsregressor.fit_predict(data=data, cv_dates=cv_dates, *args, **kwargs)
            data_forecast["algo"] = i
            data_finale = pd.concat([data_finale, data_forecast])

        if tsregressor.indexes is None:
            id_vars = ["date_last_data", "horizon", tsregressor.date]
        else:
            id_vars = ["date_last_data", "horizon"] + tsregressor.indexes + [tsregressor.date]

        table = data_finale.pivot_table(index=id_vars, columns='algo')[self.forecast_name].reset_index()
        table[self.forecast_name] = table[data_finale.algo.unique()].mean(axis=1, skipna=True)
        table = table[id_vars + [self.forecast_name]]

        return table.reset_index(drop=True)
