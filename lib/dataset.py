import os
import pandas as pd
from tqdm import tqdm


def get_bearing_paths(root_dir):
    bearing_folders = []
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name.startswith('Bearing'):
                try:
                    bearing_number = dir_name.split('Bearing')[1]
                    bearing_major, bearing_minor = map(int, bearing_number.split('_'))
                    if 1 <= bearing_major <= 3 and 1 <= bearing_minor <= 5:
                        bearing_folders.append(os.path.join(root, dir_name))
                except ValueError:
                    # 이름 형식이 맞지 않는 폴더는 무시
                    continue
    return bearing_folders

def load_data(data_path):
    file_list = os.listdir(data_path)

    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))

    df = pd.DataFrame()
    for f in tqdm(file_list, desc=f"Load Dataset: {data_path}"):
        temp = pd.read_csv(data_path + f"/{f}")
        df = pd.concat([df, temp], axis=0)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df):
    # RUL 값 할당
    df['RUL'] = range(len(df)-1, -1, -1)

    # 최소값과 최대값 계산
    min_rul = df['RUL'].min()
    max_rul = df['RUL'].max()

    # RUL 정규화
    def normalize_rul(rul, min_rul, max_rul):
        return (rul - min_rul) / (max_rul - min_rul)
    df['Normalized_RUL'] = df['RUL'].apply(lambda x: normalize_rul(x, min_rul, max_rul))

    return df
