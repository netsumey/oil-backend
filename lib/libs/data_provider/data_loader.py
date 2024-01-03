import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class PressLoader(Dataset):

    def __init__(self, root_path, data_train_path, data_vali_path,
                 data_test_path, flag='train', size=None,
                 features='S',  target='OT', scale=True, timeenc=0,
                 freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        # self.scale = False
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        # self.data_path = data_path
        self.data_train_path = data_train_path
        self.data_vali_path = data_vali_path
        self.data_test_path = data_test_path

        self.testdata = None

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        train_data = pd.read_csv(os.path.join(self.root_path, self.data_train_path))
        vali_data = pd.read_csv(os.path.join(self.root_path, self.data_vali_path))
        test_data = pd.read_csv(os.path.join(self.root_path, self.data_test_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols_train = list(train_data.columns)
        cols_train.remove(self.target)
        cols_train.remove('date')
        df_raw_train = train_data[['date'] + cols_train + [self.target]]
        num_train = int(len(df_raw_train))

        cols_vali = list(vali_data.columns)
        cols_vali.remove(self.target)
        cols_vali.remove('date')
        df_raw_vali = vali_data[['date'] + cols_train + [self.target]]
        num_vali = int(len(df_raw_vali))

        cols_test = list(test_data.columns)
        cols_test.remove(self.target)
        cols_test.remove('date')
        df_raw_test = test_data[['date'] + cols_train + [self.target]]
        num_test = int(len(df_raw_test))

        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(df_raw)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data_train = df_raw_train.columns[1:]
            df_data_train = df_raw_train[cols_data_train]

            cols_data_vali = df_raw_vali.columns[1:]
            df_data_vali = df_raw_vali[cols_data_vali]

            cols_data_test = df_raw_test.columns[1:]
            df_data_test = df_raw_test[cols_data_test]

        elif self.features == 'S':
            df_data_train = df_raw_train[[self.target]]
            df_data_vali = df_raw_vali[[self.target]]
            df_data_test = df_raw_test[[self.target]]

        if self.scale:
            train_data = df_data_train
            self.scaler.fit(train_data.values)
            data_train = self.scaler.transform(df_data_train.values)

            vali_data = df_data_vali
            self.scaler.fit(vali_data.values)
            data_vali = self.scaler.transform(df_data_vali.values)

            test_data = df_data_test
            self.scaler.fit(test_data.values)
            data_test = self.scaler.transform(df_data_test.values)

            self.testdata = data_test
        else:
            data_train = df_data_train.values
            data_vali = df_data_vali.values
            data_test = df_data_test.values

        df_stamp_train = df_raw_train[['date']]
        df_stamp_train['date'] = pd.to_datetime(df_stamp_train.date)

        df_stamp_vali = df_raw_vali[['date']]
        df_stamp_vali['date'] = pd.to_datetime(df_stamp_vali.date)

        df_stamp_test = df_raw_test[['date']]
        df_stamp_test['date'] = pd.to_datetime(df_stamp_test.date)

        if self.timeenc == 0:
            df_stamp_train['month'] = df_stamp_train.date.apply(lambda row: row.month, 1)
            df_stamp_train['day'] = df_stamp_train.date.apply(lambda row: row.day, 1)
            df_stamp_train['weekday'] = df_stamp_train.date.apply(lambda row: row.weekday(), 1)
            df_stamp_train['hour'] = df_stamp_train.date.apply(lambda row: row.hour, 1)
            df_stamp_train = df_stamp_train.drop(['date'], 1).values

            df_stamp_vali['month'] = df_stamp_vali.date.apply(lambda row: row.month, 1)
            df_stamp_vali['day'] = df_stamp_vali.date.apply(lambda row: row.day, 1)
            df_stamp_vali['weekday'] = df_stamp_vali.date.apply(lambda row: row.weekday(), 1)
            df_stamp_vali['hour'] = df_stamp_vali.date.apply(lambda row: row.hour, 1)
            df_stamp_vali = df_stamp_vali.drop(['date'], 1).values

            df_stamp_test['month'] = df_stamp_test.date.apply(lambda row: row.month, 1)
            df_stamp_test['day'] = df_stamp_test.date.apply(lambda row: row.day, 1)
            df_stamp_test['weekday'] = df_stamp_test.date.apply(lambda row: row.weekday(), 1)
            df_stamp_test['hour'] = df_stamp_test.date.apply(lambda row: row.hour, 1)
            df_stamp_test = df_stamp_test.drop(['date'], 1).values

        elif self.timeenc == 1:
            df_stamp_train = time_features(pd.to_datetime(df_stamp_train['date'].values), freq=self.freq)
            df_stamp_train = df_stamp_train.transpose(1, 0)

            df_stamp_vali = time_features(pd.to_datetime(df_stamp_vali['date'].values), freq=self.freq)
            df_stamp_vali = df_stamp_vali.transpose(1, 0)

            df_stamp_test = time_features(pd.to_datetime(df_stamp_test['date'].values), freq=self.freq)
            df_stamp_test = df_stamp_test.transpose(1, 0)

        if self.set_type == 0:
            self.data_x = data_train
            self.data_y = data_train
            self.data_stamp = df_stamp_train
        elif self.set_type == 1:
            self.data_x = data_vali
            self.data_y = data_vali
            self.data_stamp = df_stamp_vali
        elif self.set_type == 2:
            self.data_x = data_test
            self.data_y = data_test
            self.data_stamp = df_stamp_test


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self,data):
        return self.scaler.inverse_transform(data)