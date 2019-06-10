import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


class TSRegressor(object):

    def __init__(self,
                 coef=0.5,
                 horizons=[1],
                 seasonality="auto",
                 optimizer="xgboost",
                 inner_feature_eng=True,
                 inner_feature_eng_lag=14,
                 sliding_window_step=None,
                 *args,
                 **kwargs):

        self.forecast_name = "forecast"
        self.coef = self.check_coef(coef)
        self.horizons = self.check_horizons(horizons)
        self.optimizer = self.check_optimizer(optimizer)
        self.seasonality = self.check_seasonality(seasonality)
        self.inner_feature_eng = self.check_inner_feature_eng(inner_feature_eng)
        self.sliding_window_step = self.check_sliding_window_step(sliding_window_step)
        self.inner_feature_eng_lag = self.check_inner_feature_eng_lag(inner_feature_eng_lag)

        if kwargs is not None:
            for key, value in kwargs.items():
                self.__dict__[key] = value

    list_time_step = ["day", "month", "year"]
    list_optimizer = ["lgbm", "xgboost"]

    @staticmethod
    def transform_object(data, *args, **kwargs):
        for i, column in enumerate(data.columns):
            if data[column].dtype == "object":
                data[column] = data[column].astype("category").cat.codes
        return data

    @staticmethod
    def root_function(serie, coef, *args, **kwargs):
        def f(x): return np.sign(x) * np.power(abs(x), coef)
        serie = serie.apply(f)
        return serie

    @staticmethod
    def drop_invariant_features(data, *args, **kwargs):
        for i, column in enumerate(data.columns):
            if data[column].dtype != "datetime64[ns]":
                if data[column].var(axis=0, skipna=False) == 0:
                    data = data.drop(column, axis=1)

    @staticmethod
    def generate_dates(begin_date, horizon, time_step, *args, **kwargs):
        if time_step == "day":
            time_step = "D"
        if time_step == "month":
            time_step = "MS"
        if time_step == "year":
            time_step = "YS"

        dates = pd.date_range(pd.to_datetime(begin_date), periods=horizon, freq=time_step).tolist()
        for i, date in enumerate(dates):
            dates[i] = str(dates[i])[:10]

        return dates

    @staticmethod
    def get_data_holes(data, date, target, indexes=None, *args, **kwargs):

        time_step = TSRegressor.detect_time_step(data, date)

        if time_step == "day":
            time_step = "D"
        if time_step == "month":
            time_step = "MS"
        if time_step == "year":
            time_step = "YS"

        flag = 0
        if indexes is None:
            flag = 1
            indexes = ["tsboost_indexes"]
            data["tsboost_indexes"] = 0

        tss = data.sort_values([date]).groupby(indexes)

        data_finale = pd.DataFrame()

        for i, ts in tss:
            max_date = ts[date].max()
            min_date = ts[date].min()

            all_dates = pd.date_range(start=min_date, end=max_date, freq=time_step)
            date_data = ts[date].unique()

            deltas = set(str(x)[:10] for x in all_dates) - set(str(x)[:10] for x in date_data)

            result = pd.DataFrame(deltas, columns=[date])
            result[date] = pd.to_datetime(result[date])

            for index in indexes:
                result[index] = ts[index].unique()[0]

            result[target] = np.NaN
            data_finale = pd.concat([data_finale.copy(), result])

        data_finale = data_finale.sort_values(indexes + [date])

        if flag == 1:
            data_finale.drop(["tsboost_indexes"], inplace=True, axis=1)

        return data_finale

    @staticmethod
    def detect_time_step(data, date):
        if data[date].dt.month.var() != 0:
            time_step = "year"
        if data[date].dt.month.var() != 0:
            time_step = "month"
        if data[date].dt.day.var() != 0:
            time_step = "day"

        return time_step

    @staticmethod
    def get_result(data_init, data_forecast, target, date, indexes=None, metric="mape", *args, **kwargs):

        data_forecast = data_forecast.drop("date_last_data", axis=1)

        if indexes == None:
            id_vars = [date]
        else:
            id_vars = indexes + [date]

        data_forecast = data_forecast.pivot_table(index=id_vars, columns='horizon')['forecast'].reset_index()

        data = pd.merge(data_init, data_forecast, how='inner', on=id_vars)

        data.dropna(inplace=True)

        data.sort_values(date, inplace=True)

        cols = [column for column in data.columns if type(column) == int]

        for col in cols:
            col1 = "MAE_" + str(col)
            if metric == "mape":
                data[col1] = 100 * np.abs(data[col] - data[target]) / data[target]
            if metric == "mae":
                data[col1] = np.abs(data[col] - data[target])
            if metric == "rmse":
                data[col1] = np.square(data[col] - data[target])

        cols = [column for column in data.columns if "MAE" in str(column)]
        table = pd.melt(data[id_vars + cols], var_name="horizon", value_name="forecast", id_vars=id_vars)
        table["horizon"] = table["horizon"].str.extract('(\d+)').astype(int)
        table = pd.pivot_table(table, values='forecast', index=['horizon'], aggfunc=np.mean)

        if metric == "rsme":
            table["forecast"] = table["forecast"]**1. / 2.

        return table

    def check_inner_feature_eng(self, inner_feature_eng):
        if (inner_feature_eng == False) or (inner_feature_eng == True):
            pass
        else:
            raise Exception('inner_feature_eng must be set at True or False')
        return inner_feature_eng

    def check_coef(self, coef):
        if (type(coef) == int) or (type(coef) == float):
            if coef > 0:
                pass
            else:
                raise Exception('coef must positive')
        else:
            raise Exception('coef must be int or float')
        return float(coef)

    def check_horizons(self, horizons):
        if type(horizons) == int:
            if horizons > 0:
                horizons = [i for i in range(1, horizons + 1)]
        if type(horizons) == list:
            for i, horizon in enumerate(horizons):
                if type(horizon) == int:
                    if horizon > 0:
                        pass
                    else:
                        raise Exception('horizons must be positives')
                else:
                    raise Exception('horizons in list must be postive integers')
        else:
            raise Exception('horizon must be positive integer or a list of postive integers')
        return horizons

    def check_optimizer(self, optimizer):
        if optimizer in TSRegressor.list_optimizer:
            if optimizer == "xgboost":
                optimizer = xgb.XGBRegressor()
            if optimizer == "lgbm":
                optimizer = lgb.LGBMRegressor()

        if isinstance(optimizer, xgb.XGBRegressor):
            return optimizer
        if isinstance(optimizer, lgb.LGBMRegressor):
            return optimizer
        else:
            raise Exception('Wrong optimizer parameter : must be "lgbm" or "xgboost" or instances of lightgbm.LGBMRegressor / xgboost.XGBRegressor')

    def check_seasonality(self, seasonality):
        if (seasonality == None) or (seasonality == "auto") or ((type(seasonality) == int) and (seasonality > 0)):
            return seasonality
        else:
            raise Exception('Seasonality must be "auto" or None or positive integer')

    def check_time_step(self, time_step):
        if time_step in TSRegressor.list_time_step:
            return time_step
        else:
            raise Exception('Wrong time step value : must be "day", "month" or "year"')

    def check_indexes(self, indexes, data):
        if indexes == None:
            data["tsboost_indexes"] = 0
            indexes = ["tsboost_indexes"]
            return indexes
        if type(indexes) != list:
            raise Exception('indexes must be a list of column name')
        for column in indexes:
            if column not in data.columns:
                raise Exception('index column "{}" is not present in data'.format(column))
        return indexes

    def check_inner_feature_eng_lag(self, inner_feature_eng_lag):
        if type(inner_feature_eng_lag) == int:
            if inner_feature_eng_lag > 0:
                return inner_feature_eng_lag
            else:
                raise Exception('inner_feature_eng_lag "{}" must be > 0'.format(inner_feature_eng_lag))
        else:
            raise Exception('inner_feature_eng_lag "{}" must be int'.format(inner_feature_eng_lag))

    def check_target(self, target, data):
        if target in data.columns:
            return target
        else:
            raise Exception('target column "{}" is not present in data'.format(target))

    def check_date(self, date, data):
        if date in data.columns:
            pass
        else:
            raise Exception('date column "{}" is not present in data'.format(date))
        if data[date].dtype == "datetime64[ns]":
            pass
        else:
            raise Exception('date column "{}" has to be of type "datetime64[ns]"'.format(date))
        return date

    def check_sliding_window_step(self, sliding_window_step):
        if sliding_window_step == None:
            return sliding_window_step
        if type(sliding_window_step) == int:
            if sliding_window_step > 0:
                return sliding_window_step
            else:
                raise Exception('sliding_window_step "{}" must be > 0'.format(sliding_window_step))
        else:
            raise Exception('sliding_window_step "{}" must be int'.format(sliding_window_step))

    def drop_date_columns(self, data, *args, **kwargs):
        for i, column in enumerate(data.columns):
            if data[column].dtype == "datetime64[ns]" and column != self.date:
                data = data.drop(column, axis=1)
        return data

    def adjust_cv_dates(self, cv_dates, time_step, horizon):

        for i, date in enumerate(cv_dates):
            if time_step == "day":
                cv_dates[i] = str(pd.to_datetime(date) - pd.DateOffset(days=horizon))[:10]
            if time_step == "month":
                cv_dates[i] = str(pd.to_datetime(date) - pd.DateOffset(months=horizon))[:10]
            if time_step == "year":
                cv_dates[i] = str(pd.to_datetime(date) - pd.DateOffset(years=horizon))[:10]

        return cv_dates

    def get_regressors(self):
        regressors = []

        for i, horizon in enumerate(self.horizons):

            if (self.seasonality == "auto"):
                if self.time_step == "day":
                    seasonality = 7
                if self.time_step == "month":
                    seasonality = 12
                if self.time_step == "year":
                    seasonality = 1
                stationarizer_lag = int(seasonality * np.ceil(float(horizon) / float(seasonality)))

            if (self.seasonality != "auto") & (self.seasonality != None):
                seasonality = self.seasonality
                stationarizer_lag = int(seasonality * np.ceil(float(horizon) / float(seasonality)))

            if (self.seasonality == None):
                stationarizer_lag = None

            regressors.append(_Regressor(horizon, stationarizer_lag, **self.__dict__))

        return regressors

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise Exception('input data must be a pandas.DataFrame')

    def check_prod_mode_forced(self, prod_mode_forced):
        if prod_mode_forced in [True, False]:
            return prod_mode_forced
        else:
            raise Exception('prod_mode_forced must be True or False')

    def fit_predict(self, data, target, date, indexes=None, cv_dates=[None], prod_mode_forced=False):

        self.check_data(data)
        self.target = self.check_target(target, data)
        self.date = self.check_date(date, data)
        self.indexes = self.check_indexes(indexes, data)
        self.time_step = self.detect_time_step(data, self.date)
        self.prod_mode_forced = self.check_prod_mode_forced(prod_mode_forced)
        self.regressors = self.get_regressors()

        miss_data = self.get_data_holes(data, self.date, self.target, self.indexes)
        data = pd.concat([data, miss_data], sort=False)
        data = data.sort_values(self.indexes + [self.date]).reset_index(drop=True)
        data[self.target] = TSRegressor.root_function(data[self.target], self.coef)
        data = self.drop_date_columns(data.copy())

        data_finale = pd.DataFrame()

        for i, regressor in enumerate(self.regressors):

            data_origin = data.copy()
            data_ml = regressor.ml_pre_process(data.copy())

            if cv_dates == [None]:
                dates = [data[data[self.target].notnull()][self.date].max()]
            else:
                if self.prod_mode_forced == True:
                    dates = cv_dates[:]
                if self.prod_mode_forced == False:
                    dates = self.adjust_cv_dates(cv_dates[:], regressor.time_step, regressor.horizon)

            for date in dates:
                data_forecast = regressor.fit_predict(data_origin.copy(), data_ml.copy(), date)
                data_finale = pd.concat([data_finale.copy(), data_forecast.copy()])

        if self.indexes == ["tsboost_indexes"]:
            data_finale.drop(["tsboost_indexes"], inplace=True, axis=1)
            self.indexes = None

        del self.regressors

        return data_finale.reset_index(drop=True)

    def get_inner_lag_feature_eng(self, data, variable):
        lags = [i for i in range(1, self.inner_feature_eng_lag + 1)]
        variables = [variable]

        for i, variable in enumerate(variables):
            for j, lag in enumerate(lags):
                column = variable + "_m-" + str(lag)
                data[column] = data.sort_values([self.date]).groupby(self.indexes)[variable].shift(lag)

        return data

    def get_inner_date_feature_eng(self, data):

        data["month"] = data[self.date].dt.month
        data["year"] = data[self.date].dt.year

        if self.time_step == "day":
            data["day"] = data[self.date].dt.day
            data["dayofw"] = data[self.date].dt.dayofweek
            data["dayofy"] = data[self.date].dt.dayofyear
            data["week"] = data[self.date].dt.week
            data["quarter"] = data[self.date].dt.quarter

        return data


class _Regressor(TSRegressor):

    def __init__(self,
                 horizon,
                 stationarizer_lag,
                 *args,
                 **kwargs):

        self.horizon = horizon
        self.stationarizer_lag = stationarizer_lag
        super(_Regressor, self).__init__(*args, **kwargs)

    def ml_pre_process(self, data):

        data = TSRegressor.transform_object(data.copy())
        data_origin = data.copy()

        if self.stationarizer_lag is not None:
            data["init" + self.target] = data[self.target]
            data = self.stationarize(data.copy(), data_origin.copy())

        if self.inner_feature_eng is True:
            data = self.get_inner_date_feature_eng(data.copy())
            data = self.get_inner_lag_feature_eng(data.copy(), self.target)

            if self.stationarizer_lag is not None:
                data = self.get_inner_lag_feature_eng(data.copy(), "init" + self.target)

        data = self.create_target(data.copy())

        return data

    def fit_predict(self, data_origin, data, cv_dates):

        data_train, data_test = self.train_test(data.copy(), self.horizon, cv_dates)

        data_test[self.forecast_name] = self.compute_forecast(data_train.copy(), data_test.copy())

        if self.stationarizer_lag is not None:
            data_test = self.unstationarize(data_test.copy(), self.forecast_name, data_origin.copy())

        data_origin[self.forecast_name] = TSRegressor.root_function(data_test[self.forecast_name], 1. / self.coef)

        cols = [self.date] + self.indexes + [self.forecast_name]

        return self.flatten_output(data_origin[data_origin[self.date] == cv_dates][cols])

    def compute_forecast(self, data, data_test):

        data = data[data.target.notnull()]

        Y_pred_test = pd.DataFrame()
        Y_pred_test["indexes"] = np.NaN
        Y_pred_test["target_predicted"] = np.NaN
        X_train = data.drop(["target"], axis=1)
        Y_train = data.target
        X_test = data_test.drop(["target"], axis=1)

        self.optimizer.fit(X_train, Y_train)
        Y_pred_test["indexes"] = X_test.index
        Y_pred_test["target_predicted"] = self.optimizer.predict(X_test)
        Y_pred_test.set_index("indexes", inplace=True)

        return Y_pred_test["target_predicted"]

    def flatten_output(self, data):

        id_vars = self.indexes + [self.date]
        data["horizon"] = self.horizon
        data["date_last_data"] = data[self.date]

        if self.time_step == "day":
            data[self.date] = data.apply(lambda v: pd.DateOffset(days=int(v['horizon'])) + v[self.date], axis=1)
        if self.time_step == "month":
            data[self.date] = data.apply(lambda v: pd.DateOffset(months=int(v['horizon'])) + v[self.date], axis=1)
        if self.time_step == "year":
            data[self.date] = data.apply(lambda v: pd.DateOffset(years=int(v['horizon'])) + v[self.date], axis=1)

        return data[["date_last_data", "horizon"] + id_vars + [self.forecast_name]]

    def train_test(self, data, horizon, test_date):

        data_test = data[data[self.date] == test_date]

        if self.time_step == "day":
            data = data[data[self.date] + pd.DateOffset(days=horizon) <= test_date]

            if self.sliding_window_step != None:
                data = data[data[self.date] + pd.DateOffset(days=horizon) + pd.DateOffset(days=self.sliding_window_step) >= test_date]

        if self.time_step == "month":
            data = data[data[self.date] + pd.DateOffset(months=horizon) <= test_date]

            if self.sliding_window_step != None:
                data = data[data[self.date] + pd.DateOffset(months=horizon) + pd.DateOffset(months=self.sliding_window_step) >= test_date]

        if self.time_step == "year":
            data = data[data[self.date] + pd.DateOffset(years=horizon) <= test_date]

            if self.sliding_window_step != None:
                data = data[data[self.date] + pd.DateOffset(years=horizon) + pd.DateOffset(years=self.sliding_window_step) >= test_date]

        data = data.drop(self.date, axis=1)
        data_test = data_test.drop(self.date, axis=1)

        return data, data_test

    def unstationarize(self, data, label, data_origin):
        data[label] = data[label] + data_origin.sort_values([self.date]).groupby(self.indexes)[self.target].shift(self.stationarizer_lag - self.horizon)
        return data

    def stationarize(self, data, data_origin):
        data[self.target] = data[self.target] - data_origin.sort_values([self.date]).groupby(self.indexes)[self.target].shift(self.stationarizer_lag)
        return data

    def create_target(self, data):
        data['target'] = data.sort_values([self.date]).groupby(self.indexes)[self.target].shift(-self.horizon)
        return data
