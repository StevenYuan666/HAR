import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import scipy.stats as stats
from sklearn.model_selection import train_test_split


def get_frames(df, frame_size=80, hop_size=40, n_features=6):
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        a_x = df['a_x'].values[i: i + frame_size]
        a_y = df['a_y'].values[i: i + frame_size]
        a_z = df['a_z'].values[i: i + frame_size]
        g_x = df['g_x'].values[i: i + frame_size]
        g_y = df['g_y'].values[i: i + frame_size]
        g_z = df['g_z'].values[i: i + frame_size]

        # Retrive the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([a_x, a_y, a_z, g_x, g_y, g_z])
        labels.append(label)
    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, n_features)
    labels = np.asarray(labels)

    return frames, labels


def get_phone_data():
    path_phone_accel = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/phone/accel/"
    files_phone_accel = os.listdir(path_phone_accel)
    path_phone_gyro = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/phone/gyro/"
    files_phone_gyro = os.listdir(path_phone_gyro)
    files_phone_accel.sort()
    files_phone_gyro.sort()
    files_phone_accel = files_phone_accel[1:]
    files_phone_gyro = files_phone_gyro[1:]
    # phone accelerator
    df_phone_accel = pd.DataFrame()
    for file in files_phone_accel:
        temp_df = pd.read_csv(path_phone_accel + file, sep=",", header=None)
        temp_df.columns = ["id", "actCode", "Timestamp", "a_x", "a_y", "a_z"]
        temp_df['a_z'] = temp_df['a_z'].str.replace(';', '')
        df_phone_accel = pd.concat([df_phone_accel, temp_df], sort=False)
    # phone gyroscope
    df_phone_gyro = pd.DataFrame()
    for file in files_phone_gyro:
        temp_df = pd.read_csv(path_phone_gyro + file, sep=",")
        temp_df.columns = ["id", "actCode", "Timestamp", "g_x", "g_y", "g_z"]
        temp_df['g_z'] = temp_df['g_z'].str.replace(';', '')
        df_phone_gyro = pd.concat([df_phone_gyro, temp_df], sort=False)
    # Merging the datasets
    df_phone_gyro = df_phone_gyro.drop("id", axis=1)
    df_phone_gyro = df_phone_gyro.drop("actCode", axis=1)
    df_phone = pd.merge(df_phone_accel, df_phone_gyro)
    df_phone.drop("id", axis=1)
    df_phone['a_x'] = df_phone['a_x'].astype('float64')
    df_phone['a_y'] = df_phone['a_y'].astype('float64')
    df_phone['a_z'] = df_phone['a_z'].astype('float64')
    df_phone['g_x'] = df_phone['g_x'].astype('float64')
    df_phone['g_y'] = df_phone['g_y'].astype('float64')
    df_phone['g_z'] = df_phone['g_z'].astype('float64')
    # Standardize
    label = LabelEncoder()
    df_phone['label'] = label.fit_transform(df_phone['actCode'])
    x = df_phone[['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']]
    y = df_phone['label']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    df_phone = pd.DataFrame(data=x, columns=['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z'])
    df_phone['label'] = y.values
    x, y = get_frames(df=df_phone)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0, stratify=y)
    x_train = x_train.reshape(61817, 1, 80, 6)
    x_test = x_test.reshape(10909, 1, 80, 6)
    return x_train, x_test, y_train, y_test


def get_watch_data():
    path_watch_accel = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/watch/accel/"
    files_watch_accel = os.listdir(path_watch_accel)
    path_watch_gyro = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/watch/gyro/"
    files_watch_gyro = os.listdir(path_watch_gyro)
    files_watch_accel.sort()
    files_watch_gyro.sort()
    files_watch_accel = files_watch_accel[1:]
    files_watch_gyro = files_watch_gyro[1:]
    # smartwatch accelerator
    df_watch_accel = pd.DataFrame()
    for file in files_watch_accel:
        temp_df = pd.read_csv(path_watch_accel + file, sep=",")
        temp_df.columns = ["id", "actCode", "Timestamp", "a_x", "a_y", "a_z"]
        temp_df['a_z'] = temp_df['a_z'].str.replace(';', '')
        df_watch_accel = pd.concat([df_watch_accel, temp_df], sort=False)
    # smartwatch gyroscope
    df_watch_gyro = pd.DataFrame()
    for file in files_watch_gyro:
        temp_df = pd.read_csv(path_watch_gyro + file, sep=",")
        temp_df.columns = ["id", "actCode", "Timestamp", "g_x", "g_y", "g_z"]
        temp_df['g_z'] = temp_df['g_z'].str.replace(';', '')
        df_watch_gyro = pd.concat([df_watch_gyro, temp_df], sort=False)
    # merging the data set
    df_watch_gyro = df_watch_gyro.drop("id", axis=1)
    df_watch_gyro = df_watch_gyro.drop("actCode", axis=1)
    df_watch = pd.merge(df_watch_accel, df_watch_gyro)
    df_watch = df_watch.drop("id", axis=1)
    df_watch['a_x'] = df_watch['a_x'].astype('float64')
    df_watch['a_y'] = df_watch['a_y'].astype('float64')
    df_watch['a_z'] = df_watch['a_z'].astype('float64')
    df_watch['g_x'] = df_watch['g_x'].astype('float64')
    df_watch['g_y'] = df_watch['g_y'].astype('float64')
    df_watch['g_z'] = df_watch['g_z'].astype('float64')
    label = LabelEncoder()
    df_watch['label'] = label.fit_transform(df_watch['actCode'])
    x = df_watch[['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']]
    y = df_watch['label']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    df_watch = pd.DataFrame(data=x, columns=['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z'])
    df_watch['label'] = y.values
    x, y = get_frames(df=df_watch)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0, stratify=y)
    x_train = x_train.reshape(71628, 1, 80, 6)
    x_test = x_test.reshape(12641, 1, 80, 6)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_phone_data()
    print(x_train.shape, x_test.shape)
    print(x_train[0].shape, x_test[0].shape)
    print(x_train[0])
    '''
    x_train, x_test, y_train, y_test = get_watch_data()
    print(x_train.shape, x_test.shape)
    print(x_train[0].shape, x_test[0].shape)
    '''
