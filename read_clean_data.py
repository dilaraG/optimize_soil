import pandas as pd
import numpy as np


file_txt = '../data/Красногорское_С1.txt'

def clean_line(line):
    return line.strip()

with open(file_txt, 'r', encoding='utf-8') as file:
    lines = file.readlines()

columns = clean_line(lines[0][4:]).replace('"', '').split()
data = [clean_line(line[:]).split() for line in lines[1:]]

df = pd.DataFrame(data = data,columns=columns)
df.iloc[:,0] = df.iloc[:,0].str.replace('"','', regex=False)

columns_to_convert = df.columns[1:]
# df.loc[:, columns_to_convert] = df.loc[:, columns_to_convert].apply(pd.to_numeric, errors='coerce')
df[columns_to_convert] = df[columns_to_convert].astype(float)

# print(df.info())
df = df[~(df[columns_to_convert] <= -1).any(axis=1)].dropna()

df = df.loc[df['ACTNUM_GDM'] != 0]
df = df.loc[df['FWL_GDM'] != 0]
df = df.loc[df['PC'] >= 0.01]
df = df.loc[df['FWL_GDM'] >= 3] 
df = df.loc[df['Кнг_W'] != 0 ]

df.loc[df["Кнг_W"] > 1, "Кнг_W"] = df["Кнг_W"] / 100

df.to_csv("../data/df_wells_krasnogor_C1.csv", index=False)


# Регионы насыщенности в список
pvtnum_list = list(df.PVTNUM_GDM.unique())
pvtnum_list.sort()


file_J = '../data/накоп_Красногорское_С2.xlsx'
df_J_0 = pd.read_excel(file_J)
df_J_0 = df_J_0.rename(columns=lambda x: x.replace('\n', ''))
df_J_0.to_csv("../data/df_production_krasnogor_C2.csv", index=False)
