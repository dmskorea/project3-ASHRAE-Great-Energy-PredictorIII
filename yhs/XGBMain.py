import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import gc, math

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import shap

for dirname, _, filenames in os.walk('../Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

building_metadata = pd.read_csv("../Data/building_metadata.csv")
sample_submission = pd.read_csv("../Data/sample_submission.csv")
test = pd.read_csv("../Data/test.csv")
train = pd.read_csv("../Data/train.csv")
weather_test = pd.read_csv("../Data/weather_test.csv")
weather_train = pd.read_csv("../Data/weather_train.csv")


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

metadata_df = reduce_mem_usage(building_metadata)
train_df = reduce_mem_usage(train)
test_df = reduce_mem_usage(test)
weather_train_df = reduce_mem_usage(weather_train)
weather_test_df = reduce_mem_usage(weather_test)

train_metadata_df = train_df.join(other=metadata_df.set_index('building_id'), on='building_id', how='left', lsuffix='_l', rsuffix='_r')
full_train_df = train_metadata_df.merge(right=weather_train_df, how='left')

del train_df
del weather_train_df
del train_metadata_df

full_test_df = test_df.merge(metadata_df, on='building_id', how='left')
full_test_df = full_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

del metadata_df
del weather_test_df
del test_df

def mean_without_overflow_fast(col):
    col /= len(col)
    return col.mean() * len(col)

missing_values = (100-full_train_df.count() / len(full_train_df) * 100).sort_values(ascending=False)
missing_features = full_train_df.loc[:, missing_values > 0.0]
missing_features = missing_features.apply(mean_without_overflow_fast)

for key in full_train_df.loc[:, missing_values > 0.0].keys():
    if key == 'year_built' or key == 'floor_count':
        full_train_df[key].fillna(math.floor(missing_features[key]), inplace=True)
        full_test_df[key].fillna(math.floor(missing_features[key]), inplace=True)
    else:
        full_train_df[key].fillna(missing_features[key], inplace=True)
        full_test_df[key].fillna(missing_features[key], inplace=True)

full_train_df["timestamp"] = pd.to_datetime(full_train_df["timestamp"])
full_test_df["timestamp"] = pd.to_datetime(full_test_df["timestamp"])

def transform(df):
    df['hour'] = np.uint8(df['timestamp'].dt.hour)
    df['day'] = np.uint8(df['timestamp'].dt.day)
    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
    df['month'] = np.uint8(df['timestamp'].dt.month)
    df['year'] = np.uint8(df['timestamp'].dt.year - 1900)

    df['square_feet'] = np.log(df['square_feet'])

    return df

full_train_df = transform(full_train_df)
full_test_df = transform(full_test_df)

dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
full_train_df['is_holiday'] = (full_train_df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
full_test_df['is_holiday'] = (full_test_df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

full_train_df.loc[(full_train_df['weekday'] == 5) | (full_train_df['weekday'] == 6) , 'is_holiday'] = 1
full_test_df.loc[(full_test_df['weekday']) == 5 | (full_test_df['weekday'] == 6) , 'is_holiday'] = 1

full_test_df = full_test_df.drop(['timestamp'], axis=1)
full_train_df = full_train_df.drop(['timestamp'], axis=1)
print (f'Shape of training dataset: {full_train_df.shape}')
print (f'Shape of testing dataset: {full_test_df.shape}')

full_train_df = reduce_mem_usage(full_train_df)
full_test_df = reduce_mem_usage(full_test_df)

def degToCompass(num):
    val=int((num/22.5)+.5)
    arr=[i for i in range(0,16)]
    return arr[(val % 16)]

full_train_df['wind_direction'] = full_train_df['wind_direction'].apply(degToCompass)

beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9),
          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

for item in beaufort:
    full_train_df.loc[(full_train_df['wind_speed']>=item[1]) & (full_train_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]


le = LabelEncoder()
full_train_df['primary_use'] = le.fit_transform(full_train_df['primary_use'])

categoricals = ['site_id', 'building_id', 'primary_use', 'hour', 'weekday', 'meter',  'wind_direction', 'is_holiday']
drop_cols = ['sea_level_pressure', 'wind_speed']
numericals = ['square_feet', 'year_built', 'air_temperature', 'cloud_coverage',
              'dew_temperature', 'precip_depth_1_hr', 'floor_count', 'beaufort_scale']

feat_cols = categoricals + numericals


target = np.log1p(full_train_df["meter_reading"])
del full_train_df["meter_reading"]
full_train_df = full_train_df.drop(drop_cols, axis = 1)

full_test_df.to_pickle('full_test_df.pkl')
del full_test_df

gc.collect()

full_train_df = reduce_mem_usage(full_train_df)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

scores = []
params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.08,
    'max_depth': 10,
    'subsample': 1,
    'objective': 'reg:linear',
    'eval_metric': 'mlogloss',
    'min_child_weight': 3,
    'gamma': 0.25,
    'n_estimators': 500
}

kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

models = []

best_xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,gamma=0,learning_rate=0.07, max_depth=7, min_child_weight=1.5, n_estimators=400, reg_alpha=0.75, reg_lambda=0.45,subsample=0.6, random_state=7, tree_method='gpu_exact')
best_xgb_model.fit(full_train_df[feat_cols],target)

full_test_df = pd.read_pickle('full_test_df.pkl')

full_test_df["primary_use"] = le.transform(full_test_df["primary_use"])
full_test_df['wind_direction'] = full_test_df['wind_direction'].apply(degToCompass)

for item in beaufort:
    full_test_df.loc[(full_test_df['wind_speed']>=item[1]) & (full_test_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

res = np.expm1(best_xgb_model.predict(full_test_df[feat_cols]))

submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
submission.to_csv('submission_fe_lgbm.csv', index=False)
submission
