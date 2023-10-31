import pandas as pd
import numpy as np

def get_data_with_filter(data:pd.DataFrame, column_name:str, filter_type:str, filter_num:int) -> pd.DataFrame:
    filter_type_list = ['ge', 'gt', 'le', 'lt']
    assert filter_type in filter_type_list
    output = data[data[column_name].__getattribute__(filter_type)(filter_num)]
    return output


def get_unique_ratio(path, start_column_idx=1):
    data = pd.read_csv(path).iloc[:, start_column_idx:]
    occurance_num = np.sum(data.values, axis=0)
    output = {}
    for idx, c in enumerate(data.columns):
        temp = data[c][data[c] > 0].shape[0]
        output[c] = temp/occurance_num[idx]
    return output
