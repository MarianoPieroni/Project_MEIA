import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import load

def load_processed_dat():

    
    df=pd.read_csv('cleaned_data.csv')
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")    
    Info=load('Info.joblib')
    print(f"Generos disponiveis: {Info['genres']}")
    return df,Info
       
def prepare_data_for_clustering(df, genres):
  
    features = ['rating'] + genres
    
    # Verificar se todas as features existem
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Features nao encontradas {missing_features}")
        features = [f for f in features if f in df.columns]
    
    # Criar dataframe apenas com essas features
    cluster_data = df[features].copy()
    
    print(f"Features utilizadas para clustering: {features}")
    print(f"Dimensoes dos dados: {cluster_data.shape}")
    
    # Normalizar os dados
    scaler = StandardScaler()
    
    # Aplicar scaling apenas nas colunas numéricas
    numeric_cols = ['rating']
    
    # Verificar se as colunas existem
    available_numeric = [col for col in numeric_cols if col in cluster_data.columns]
    
    if available_numeric:
        print(f"Aplicando scaling nas colunas: {available_numeric}")
        cluster_data[available_numeric] = scaler.fit_transform(cluster_data[available_numeric])
    else:
        print("Aviso: Nenhuma coluna numérica disponível para scaling")
    
    # Para os gêneros (que são binários 0/1), garantir que são inteiros
    for genre in genres:
        if genre in cluster_data.columns:
            cluster_data[genre] = cluster_data[genre].astype(int)
        
    return cluster_data, scaler

def elbow_method(data, max_clusters=10):
 
    # Lista para armazenar as inércias
    inertias = []
    
    # Testar diferentes números de clusters
    cluster_range = range(1, max_clusters + 1)
    
    for k in cluster_range:
        # Criar modelo KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)

        # Treinar o modelo
        kmeans.fit(data)
        
        # Armazenar a inércia (soma das distâncias quadradas)
        inertias.append(kmeans.inertia_)
        
        print(f"  Clusters: {k:2d}, Inercia: {kmeans.inertia_:.2f}")
    
    return inertias, cluster_range

def plot_elbow_method(inertias, cluster_range, save_fig=True):

    fig=plt.subplots(figsize=(10, 5))
    plt.plot(cluster_range, inertias, 'bo-')
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # Carregar dados processados
    df_clean, basic_info = load_processed_dat()
    
    if df_clean is not None and basic_info is not None:
        genres = basic_info['genres']
        
        # Preparar dados para clustering
        cluster_data, scaler = prepare_data_for_clustering(df_clean, genres)
        
        # Aplicar método do cotovelo
        inertias, cluster_range = elbow_method(cluster_data, max_clusters=15)
        
        # Plotar gráfico do cotovelo
        elbow_points = plot_elbow_method(inertias, cluster_range, save_fig=True)