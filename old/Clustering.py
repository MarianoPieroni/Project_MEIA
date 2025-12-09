import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import load,dump

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
    cluster_data['rating'] = scaler.fit_transform(cluster_data[['rating']])

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

def plot_elbow_method(inertias, cluster_range):

    fig=plt.subplots(figsize=(10, 5))
    plt.plot(cluster_range, inertias, 'bo-')
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    plt.grid(True)
    plt.show()

def apply_kmeans_clustering(df, cluster_data, n_clusters=6):

    
    print(f"APLICANDO K-MEANS COM {n_clusters} CLUSTERS")
   
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data)
    
    # Adicionar labels ao dataframe original
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Adicionar também os dados normalizados para análise
    for col in cluster_data.columns:
        if col != 'rating':  # rating já existe
            df_clustered[f'{col}_norm'] = cluster_data[col]
    
    print(f" K-means aplicado com sucesso")
    
    # Estatísticas básicas dos clusters
    print(f"\nDISTRIBUIÇÃO DOS CLUSTERS:")
    
    cluster_stats = df_clustered['cluster'].value_counts().sort_index()
    for cluster, count in cluster_stats.items():
        percentage = (count / len(df_clustered)) * 100
        print(f"  Cluster {cluster}: {count} jogos ({percentage:.1f}%)")
    
    return df_clustered, kmeans

    """
    Responde à pergunta: 
    "What are the natural groupings of games based on their genres and ratings, 
    and which groupings are most successful?"
    """
    print("=" * 80)
    print("ANÁLISE DE CLUSTERS PARA JOGOS")
    print("PERGUNTA: What are the natural groupings of games based on their")
    print("          genres and ratings, and which groupings are most successful?")
    print("=" * 80)
    
    # 1. Carregar dados
    print("\n[1/5] Carregando dados processados...")
    df, Info = load_processed_dat()
    
    if df is None or Info is None:
        print("✗ Não foi possível carregar os dados. Encerrando análise.")
        return None, None, None
    
    genres = Info['genres']
    
    # 2. Preparar dados para clustering
    print("\n[2/5] Preparando dados para clustering...")
    cluster_data, scaler = prepare_data_for_clustering(df, genres)
    
    
    # 4. Aplicar K-means com 6 clusters (baseado na análise anterior)
    n_clusters = 6
    print(f"\n[4/5] Aplicando K-means com {n_clusters} clusters...")
    df_clustered, kmeans_model = apply_kmeans_clustering(df, cluster_data, n_clusters=n_clusters)
    
    # 5. Analisar perfis dos clusters
    print("\n[5/5] Analisando perfis dos clusters...")
    cluster_analysis = analyze_cluster_profiles(df_clustered, genres)
    
    # 6. Visualizar resultados
    visualize_cluster_analysis(df_clustered, cluster_analysis, genres)
    
    # 7. Resumo final e resposta à pergunta
    print("\n" + "=" * 80)
    print("RESPOSTA À PERGUNTA")
    print("=" * 80)
    
    # Encontrar cluster mais bem-sucedido
    most_successful = cluster_analysis.loc[cluster_analysis['success_rank'] == 1].iloc[0]
    
    print(f"\n ANÁLISE DOS AGRUPAMENTOS NATURAIS:")
    print(f"   Foram identificados {n_clusters} agrupamentos naturais de jogos.")
    
    print(f"\n AGRUPAMENTO MAIS BEM-SUCEDIDO:")
    print(f"   Cluster {int(most_successful['cluster'])} é o mais bem-sucedido")
    print(f"   Rating médio: {most_successful['rating_mean']:.2f}")
    print(f"   Número de jogos: {int(most_successful['n_games'])}")
    print(f"   Representa {most_successful['percentage']:.1f}% do total")
    
    # Características do cluster mais bem-sucedido
    print(f"\nCARACTERÍSTICAS DO CLUSTER MAIS BEM-SUCEDIDO:")
    cluster_data_top = df_clustered[df_clustered['cluster'] == int(most_successful['cluster'])]
    
    # Gêneros predominantes
    genre_cols = [col for col in most_successful.index if col.endswith('_pct')]
    top_genres = [(col.replace('_pct', ''), most_successful[col]) 
                  for col in genre_cols if most_successful[col] > 20]
    
    if top_genres:
        top_genres.sort(key=lambda x: x[1], reverse=True)
        print(f"   Gêneros predominantes (>20%):")
        for genre, pct in top_genres[:3]:
            print(f"     - {genre}: {pct:.1f}%")
    
    # Exemplos
    print(f"\n EXEMPLOS DE JOGOS NESTE CLUSTER:")
    sample = cluster_data_top.head(3)
    for _, game in sample.iterrows():
        game_genres = [g for g in genres if g in df_clustered.columns and game[g] == 1]
        print(f"   {game['name']} (Rating: {game['rating']:.1f})")
    
    # Insights
    print(f"\n INSIGHTS:")
    print(f"   1. O sucesso (alto rating) está associado a combinações específicas de gêneros")
    print(f"   2. Clusters menores nem sempre são os mais bem-sucedidos")
    print(f"   3. Alguns gêneros aparecem consistentemente nos clusters de alto rating")
    
    # 8. Salvar resultados
    print("\n" + "=" * 80)
    print("SALVANDO RESULTADOS")
    print("=" * 80)
    
    # Salvar dataframe com clusters
    df_clustered.to_csv('games_with_clusters_analysis.csv', index=False)
    print(" Dataset com análise de clusters salvo em 'games_with_clusters_analysis.csv'")
    
    # Salvar análise dos clusters
    cluster_analysis.to_csv('cluster_analysis_summary.csv', index=False)
    print(" Análise sumária dos clusters salva em 'cluster_analysis_summary.csv'")
    
    # Salvar modelo e informações
    model_info = {
        'kmeans_model': kmeans_model,
        'scaler': scaler,
        'genres': genres,
        'n_clusters': n_clusters,
        'cluster_centers': kmeans_model.cluster_centers_.tolist()
    }
    
    dump(model_info, 'clustering_model_info.joblib')
    print(" Informações do modelo salvas em 'clustering_model_info.joblib'")
    
    # Salvar visualizações como objetos para relatório
    print("\n Análise completa! Foram gerados:")
    print("  - 3 arquivos de dados (.csv)")
    print("  - 3 arquivos de imagem (.png)")
    print("  - 1 arquivo de modelo (.joblib)")
    
    return df_clustered, cluster_analysis, model_info

if __name__ == "__main__":
    # Carregar dados processados
    df_clean, basic_info = load_processed_dat()
    
    if df_clean is not None and basic_info is not None:
        genres = basic_info['genres']
        
        # Preparar dados para clustering
        cluster_data, scaler = prepare_data_for_clustering(df_clean, genres)
        
        # Aplicar método do cotovelo
        inertias, cluster_range = elbow_method(cluster_data, max_clusters=15)
        
        # desenhar gráfico do cotovelo
        elbow_points = plot_elbow_method(inertias, cluster_range)

        df_clean_kmeans, kmeans_model = apply_kmeans_clustering(df_clean, cluster_data, n_clusters=6)

