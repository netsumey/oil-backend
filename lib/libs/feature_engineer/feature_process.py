import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics.pairwise import cosine_similarity


cls = MinMaxScaler()
scaler = StandardScaler()

class DataPretreatment:
    def __init__(self,dir_path):
        self.dir_path = dir_path

    def read_files_in_directory(self, dir_path):

        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                if item.endswith('.csv'):
                    # self._handle_files(item, item_path)
                    self.feature_selected02(item, item_path)
                    # self._missing_values_complement(item, item_path)
            elif os.path.isdir(item_path):
                if item in ['yali_test', 'yali_train', 'yali_vali']:
                    self.read_files_in_directory(item_path)

    def feature_selected(self, file_name, file_path):
        # 读取CSV文件
        if file_name == 'yali_train.csv':
            columns_to_read = lambda col: col != 'date'
            df = pd.read_csv(file_path, usecols=columns_to_read, encoding='UTF-8')

            # 转换数据为数值型
            df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'), axis=0)

            # 获取目标变量并移除
            # target_df = df['press']
            # df = df.drop('press', axis=1)
            # print(df)
            # 转换为 NumPy 数组
            df_array = np.array(df)
            standardized_data = scaler.fit_transform(df_array)

            # 计算特征方差并找到方差为零的特征索引
            # variances = np.var(df_array, axis=0)
            # zero_variance_indices = np.where(variances == 0)[0]

            # 删除方差为零的特征
            # df_array_filtered = np.delete(df_array, zero_variance_indices, axis=1)

            # 转换目标变量为 NumPy 数组
            # target_array = np.array(target_df)
            target_array = standardized_data[:, -1]
            # 使用切片选择除最后一列以外的所有列
            array_without_last_column = standardized_data[:, :-1]

            # 选择K个最好的特征，返回选择特征后的数据
            # selected_features = SelectKBest(f_regression, k=5).fit_transform(df_array_filtered, target_array)

            # 获取被选择特征的列索引
            selected_indices = SelectKBest(f_regression, k=2).fit(array_without_last_column, target_array).get_support(indices=True)

            # 获取被选择特征的列名
            print(selected_indices)
            selected_column_names = df.columns[selected_indices]

            # 打印输出选择的特征列名
            print(f"{file_name}-Selected features:", selected_column_names.values)

    def feature_selected01(self, file_name, file_path):
        # 读取CSV文件
        if file_name == 'yali_train.csv':
            columns_to_read = lambda col: col != 'date'
            df = pd.read_csv(file_path, usecols=columns_to_read, encoding='UTF-8')
            correlation_matrix = df.corr().abs()['press'].sort_values(ascending=False)
            print("特征与目标变量的皮尔逊相关系数:")
            print(correlation_matrix)

    def feature_selected02(self, file_name, file_path):
        # 读取CSV文件
        if file_name == 'yali_train.csv':
            columns_to_read = lambda col: col != 'date'
            df = pd.read_csv(file_path, usecols=columns_to_read, encoding='UTF-8')
            # 计算特征之间的余弦相似度矩阵
            cosine_sim_matrix = cosine_similarity(df.values.T)
            # 将结果转换为 DataFrame（可选）
            cosine_sim_df = pd.DataFrame(cosine_sim_matrix, columns=df.columns, index=df.columns)

            print("特征之间的余弦相似度相关系数矩阵:")
            print(cosine_sim_df)


    def _missing_values_complement(self, file_name, file_path):
        print('file_name', file_name)
        file_name_parts = file_name.split('.', 1)
        df = pd.read_csv(file_path, encoding='UTF-8')
        df['paichu'] = df['paichu'].fillna(method='ffill')
        df.to_csv('../dataset/YALI_fill_data/' + file_name_parts[0] +'_complement' + '.csv', index=0, encoding='utf_8_sig')

    def _handle_files(self, file_name, file_path):
        print('file_name', file_name)
        columns_to_read = lambda col: col != 'date'
        df = pd.read_csv(file_path, usecols=columns_to_read, encoding='UTF-8')
        data_array = df.values
        data_array = cls.fit_transform(data_array)
        data_mean = data_array.mean()
        data_variance = data_array.std()
        print('打印均值', data_mean)
        print('打印方差', data_variance)

if __name__ == "__main__":

    dir_path = '../dataset/YALI_origral/'
    data_pretreatment = DataPretreatment(dir_path)
    data_pretreatment.read_files_in_directory(dir_path)