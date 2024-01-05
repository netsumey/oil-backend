import argparse
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error as mmmse
from utils.metrics import metric
import matplotlib.pyplot as plt

def list_files_and_folders(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径 '{path}' 不存在")
        return

    print(f"路径 '{path}' 下的文件和文件夹：")

    # 获取路径下的所有内容
    contents = os.listdir(path)

    # 遍历所有内容并分类为文件夹和文件
    folders = []
    files = []
    for item in contents:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.append(item)
        else:
            files.append(item)

    # 打印文件夹
    print("文件夹：")
    for folder in folders:
        print(folder)

    # 打印文件
    print("\n文件：")
    for file in files:
        print(file)

def _data_process(data_path_pred, data_path_true, title, test_csv):
    df_raw = pd.read_csv(test_csv)
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    df_data = df_data.iloc[:, -1:]

    scaler = StandardScaler()
    scaler.fit(df_data)

    data_sca = scaler.transform(df_data.values)
    data_row = scaler.inverse_transform(data_sca)

    # trues_row = data_row

    preds_sca = np.load(data_path_pred)
    trues_sca = np.load(data_path_true)

    preds_sca = preds_sca[:, -1, :]
    trues_sca = trues_sca[:, -1, :]


    # preds = torch.flatten(preds_sca, start_dim=0, end_dim=1)
    preds = preds_sca.reshape(-1, preds_sca.shape[-1])
    trues = trues_sca.reshape(-1, trues_sca.shape[-1])

    preds_row = scaler.inverse_transform(preds)
    trues_row = scaler.inverse_transform(trues)

    # preds_row = preds
    # trues_row = trues
    # print('true', trues_row.shape)
    # mae, mse, rmse, mape, mspe = metric(preds_row, trues_row)
    # print('mse', mse)
    # print('mae', mae)
    a = preds_row
    b = trues_row
    plt.figure()
    plt.plot(range(len(a)), a, color='blue', label="pred")
    plt.plot(range(len(b)), b, color='red', label="true")
    plt.legend(['pred', 'true'])
    plt.title(title)

    plt.savefig('tmp.png')
    plt.show()

# # 你可以替换这里的路径为你想要检查的路径
# directory_path = './results/'
# a = 'long_term_forecast_yali_10_10_RNN_yali_ftMS_sl10_ll5_pl10_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
# # list_files_and_folders(directory_path)
# _data_process('./results/'+a+'/pred.npy',
#               './results/'+a+'/true.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calmse')

    parser.add_argument('--directory', type=str, required=False)
    parser.add_argument('--a', type=str, required=False)
    parser.add_argument('--title', type=str, required=False)
    parser.add_argument('--test_csv', type=str, required=False)

    args = parser.parse_args()

    _data_process(args.directory + "/" +  args.a + "/pred.npy",
                  args.directory + "/" + args.a + "/true.npy", args.title, args.test_csv)

