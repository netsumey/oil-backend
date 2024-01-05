import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer


class Operator:
    def __init__(self, path, out_path):
        self.path = path
        self.out_path = out_path
        self.csv = pd.read_csv(path)

    def delete_row_by_index(self, indices):
        if indices != '':
            indices = [int(idx) for idx in indices.split(',')]
            self.csv.drop(indices, inplace=True)

    def delete_col_by_index(self, col_indices):
        if col_indices != '':
            col_indices = [int(idx) for idx in col_indices.split(',')]
            self.csv.drop(self.csv.columns[col_indices], axis=1, inplace=True)


    def delete_col_by_name(self, col_names):
        if col_names != '':
            self.csv.drop(columns=col_names.split(','), inplace=True)

    def normalize_columns(self, method):
        if method == "MinMax":
            scaler = MinMaxScaler()
        elif method == "Z":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid normalization method. Use 'MinMax' or 'Z'.")

        self.csv.iloc[:, 1:] = scaler.fit_transform(self.csv.iloc[:, 1:])

    def handle_missing_values(self, method):
        if method == "Delete":
            self.csv.dropna(inplace=True)
        else:
            if method == "Mean":
                imputer = SimpleImputer(strategy="mean")
            elif method == "Median":
                imputer = SimpleImputer(strategy="median")
            elif method == "Mode":
                imputer = SimpleImputer(strategy="most_frequent")
            elif method == "Forward":
                self.csv.fillna(method="ffill", inplace=True)
                return
            elif method == "Backward":
                self.csv.fillna(method="bfill", inplace=True)
                return
            else:
                raise ValueError("Invalid missing value handling method.")

            self.csv.iloc[:, 1:] = imputer.fit_transform(self.csv.iloc[:, 1:])

    def save_changes(self):
        self.csv.to_csv(self.out_path, index=False)

    def get_row_count(self):
            return len(self.csv)

    def get_column_count(self):
        return len(self.csv.columns)

    def get_column_names(self):
        return list(self.csv.columns)

    def get_top_rows(self, k):
        return self.csv.head(k)

    def get_csv(self):
        return self.csv