import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

def EDA():

    df = pd.read_csv("imdb-videogames.csv")
    df_clean = df.copy()
    
    print("DATASET ORIGINAL")
    print(f"Dimensoes: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    return df, df_clean

def explore_dataset(df_clean):
    print("\nESTATISTICAS DESCRITIVAS")
    print(df_clean.describe(include='all'))
    
    # CLASSIFICAÇÃO DAS VARIÁVEIS
    genres = ['Action', 'Adventure', 'Comedy', 'Crime', 'Family', 
              'Fantasy', 'Mystery', 'Sci-Fi', 'Thriller']
    
    variables = {
        'Numericas': ['year', 'rating', 'votes'],
        'Categoricas': ['certificate', 'name', 'plot'],
        'Binarias': genres
    }
    # Lista de gêneros
    print("\nVARIAVEIS")
    print("Variavel Dependente (target): 'rating'")
    print("Variaveis Independentes: 'year', 'votes', 'certificate', gêneros")
    print(f"Generos disponiveis: {genres}")
    
    print(f"\nValores duplicados: {df_clean.duplicated().sum()}")
    print("\nVALORES NULOS")
    missing_data = df_clean.isnull().sum()
    missing_data_total = df_clean.isnull().sum().sum()
    print(missing_data[missing_data > 0])
    print(f"Total de valores nulos: {missing_data_total}")
    print(f"Total de linhas no dataset: {df_clean.shape[0]}")
    
    return variables,genres

def clean_data(df_clean):
    print("\nINICIANDO LIMPEZA DOS DADOS")
    
    # TRATAMENTO DE VALORES NULOS
    # votes -> substituir por zero 
    df_clean['votes'] = df_clean['votes'].str.replace(',', '').astype(float)
    df_clean['votes'] = df_clean['votes'].fillna(0)
    
    # rating -> substituir pela mediana
    df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
    
    # year - Eliminar linhas com poucos nulos
    if df_clean['year'].isna().sum() < len(df_clean) * 0.05:  # menos de 5%
        df_clean = df_clean.dropna(subset=['year'])
    else:
        df_clean['year'] = df_clean['year'].fillna(df_clean['year'].mode()[0])
    
    # certificate -> substituir por 'Not Rated'
    df_clean['certificate'] = df_clean['certificate'].fillna('Not Rated')
    
    print("\nDADOS APOS LIMPEZA")
    missing_data = df_clean.isnull().sum()
    missing_data_total = df_clean.isnull().sum().sum()
    print(missing_data[missing_data > 0])
    print(f"Total de valores nulos: {missing_data_total}")
    print(f"Total de linhas no dataset: {df_clean.shape[0]}")
    
    return df_clean

def detect_outliers(df_clean):
    def detect_outliers_iqr(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = column[(column < lower_bound) | (column > upper_bound)]
        return outliers
    
    print("\nOUTLIERS DETECTADOS")
    outliers_info = {}
    for col in ['rating', 'votes', 'year']:
        outliers = detect_outliers_iqr(df_clean[col])
        outliers_info[col] = len(outliers)
        print(f"{col}: {len(outliers)} outliers")
    
    return outliers_info

def save_cleaned_data(df_clean,genres):
    print("\nSALVANDO DADOS LIMPOS")
    
    # Salvar dados limpos
    df_clean.to_csv('cleaned_data.csv', index=False)
    
    # Salvar informações básicas
    basic_info = {
        'genres': genres,
        'original_columns': df_clean.columns.tolist(),
        'data_shape': df_clean.shape
    }
    
    with open('basic_info.pkl', 'wb') as f:
        pickle.dump(basic_info, f)
    
    print(f"Dados limpos salvos em 'cleaned_data.csv'")
    print(f"Shape dos dados limpos: {df_clean.shape}")
    print(f"Generos disponiveis: {genres}")
    
    return df_clean

def main():
    
    df, df_clean = EDA()
    
    
    variables,genres = explore_dataset(df_clean)
    
    
    df_clean = clean_data(df_clean)
    
    
    outliers_info = detect_outliers(df_clean)
    
    df_clean = save_cleaned_data(df_clean,genres)
    print(f"Dataset final: {df_clean.shape}")
    
    return df_clean

if __name__ == "__main__":
    df_clean = main()