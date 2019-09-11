# Author:Sarah Shi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_columns', None)


def data_clean():
    # 1.读取数据表
    df = pd.read_excel('ClimateChange.xlsx', 'Data')
    # 2.筛选数据
    df_gdp = df[df['Series code'] == 'NY.GDP.MKTP.CD'].set_index('Country code')
    df_co2 = df[df['Series code'] == 'EN.ATM.CO2E.KT'].set_index('Country code')
    # 3.将..替换成nan
    df_gdp.replace('..', pd.np.nan, inplace=True)
    df_co2.replace('..', pd.np.nan, inplace=True)
    # 4.填充空白数据
    df_gdp_fill = df_gdp.iloc[:, 5:].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    df_co2_fill = df_co2.iloc[:, 5:].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    # 5.得到数据总和
    df_gdp_fill['GDP-SUM'] = df_gdp_fill.sum(axis=1)
    df_co2_fill['CO2-SUM'] = df_co2_fill.sum(axis=1)
    # 6.合并gdp和co2的数据(concat合并的是列表)
    df_concat = pd.concat([df_co2_fill['CO2-SUM'],df_gdp_fill['GDP-SUM']], axis=1)
    # 7.把空白数据填充为0
    df_concat_fill = df_concat.fillna(value=0)

    return df_concat_fill


def co2_gdp_plot():
    df = data_clean()
    # 1.对数据进行归一化处理
    df_norm = (df - df.min()) / (df.max() - df.min())
    # 2.获取5个常任理事国['CHN', 'USA', 'GBR', 'FRA','RUS']在country code中的位置，并和country name一一对应
    countries = ['CHN', 'USA', 'GBR', 'FRA', 'RUS']
    countries_label = []
    countries_name = []
    for i in range(len(df_norm)):
        # 方法一if df_norm.iloc[i].name in countries:
        # 方法二
        if df_norm.index[i] in countries:
            countries_label.append(i)
            countries_name.append(df_norm.index[i])

    #values得到的是numpy.ndarray对象，所以这里要用tolist转换为list
    china = np.round(df_norm['CHN':'CHN'].values,3).tolist()[0]

    fig = plt.subplot()
    df_norm.plot(kind='line', title='GDP-CO2', ax=fig)
    plt.xlabel('Countries')
    plt.ylabel('Values')
    plt.xticks(countries_label, countries_name, rotation='vertical')
    plt.show()
    
    return fig, china

if __name__ == '__main__':
    co2_gdp_plot()

