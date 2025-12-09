import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import load, dump
from sklearn.metrics import silhouette_score, silhouette_samples
# ============================================
# 1. CARREGAR DADOS
# ============================================
def carregar_dados():
    df = pd.read_csv('cleaned_data.csv')
    print(f"Jogos carregados: {len(df)}")
    
    info = load('Info.joblib')
    generos = info['genres']
    print(f"Generos: {generos}")
    
    return df, generos

# ============================================
# 2. PREPARAR DADOS
# ============================================
def preparar_dados(df, generos):
    # Selecionar features: rating + generos
    features = ['rating'] + [g for g in generos if g in df.columns]
    
    # Criar dataset para clustering
    dados_cluster = df[features].copy()
    
    # Normalizar rating
    scaler = StandardScaler()
    dados_cluster['rating'] = scaler.fit_transform(dados_cluster[['rating']])
    
    # Garantir que generos sao inteiros
    for genero in generos:
        if genero in dados_cluster.columns:
            dados_cluster[genero] = dados_cluster[genero].astype(int)
    
    print(f"Features usadas: {len(features)}")
    return dados_cluster, scaler

# ============================================
# 3. ENCONTRAR MELHOR NUMERO DE CLUSTERS
# ============================================
def encontrar_melhor_k(dados):
    inercias = []
    k_range = range(1, 11)  # Testa de 1 a 10 clusters
    
    print("\nTestando diferentes numeros de clusters:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dados)
        inercias.append(kmeans.inertia_)
        print(f"  K={k}: Inercia={kmeans.inertia_:.2f}")
    
    # Plot do metodo do cotovelo
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inercias, 'bo-')
    plt.xlabel('Numero de Clusters (K)')
    plt.ylabel('Inercia')
    plt.title('Metodo do Cotovelo')
    plt.grid(True)
    plt.savefig('elbow_method.png', dpi=300)
    plt.show()
    
    return inercias

# ============================================
# 4. APLICAR K-MEANS
# ============================================
def aplicar_kmeans(df, dados, k=6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(dados)
    
    df_com_clusters = df.copy()
    df_com_clusters['cluster'] = clusters
    
    print(f"\nDistribuicao dos clusters:")
    for i in range(k):
        count = (df_com_clusters['cluster'] == i).sum()
        print(f"  Cluster {i}: {count} jogos")
    
    return df_com_clusters, kmeans

# ============================================
# 5. ANALISAR CLUSTERS
# ============================================
def analisar_clusters(df_com_clusters, generos):
    print("\n" + "="*50)
    print("ANALISE DOS CLUSTERS")
    print("="*50)
    
    resultados = []
    
    for cluster_id in sorted(df_com_clusters['cluster'].unique()):
        # Filtrar jogos do cluster
        cluster_df = df_com_clusters[df_com_clusters['cluster'] == cluster_id]
        
        # Estatisticas basicas
        info = {
            'cluster': cluster_id,
            'n_jogos': len(cluster_df),
            'rating_medio': cluster_df['rating'].mean(),
            'rating_max': cluster_df['rating'].max(),
            'rating_min': cluster_df['rating'].min()
        }
        
        # Percentagem de cada genero
        for genero in generos:
            if genero in cluster_df.columns:
                pct = (cluster_df[genero].sum() / len(cluster_df)) * 100
                info[f'{genero}_pct'] = pct
        
        # Encontrar generos principais (>40%)
        principais = []
        for genero in generos:
            if f'{genero}_pct' in info and info[f'{genero}_pct'] > 40:
                principais.append(genero)
        
        info['generos_principais'] = ', '.join(principais) if principais else 'Misto'
        
        resultados.append(info)
    
    # Criar DataFrame com resultados
    analise = pd.DataFrame(resultados)
    
    # Ordenar por rating medio
    analise = analise.sort_values('rating_medio', ascending=False)
    analise['ranking'] = range(1, len(analise) + 1)
    
    # Mostrar tabela resumo
    print("\nRESUMO DOS CLUSTERS:")
    print("-"*70)
    print(f"{'Cluster':<8} {'Jogos':<6} {'Rating':<8} {'Generos_Principais'}")
    print("-"*70)
    
    for _, row in analise.iterrows():
        print(f"{int(row['cluster']):<8} {int(row['n_jogos']):<6} {row['rating_medio']:<8.2f} {row['generos_principais']}")
    
    return analise

# ============================================
# 6. VISUALIZACOES
# ============================================
def criar_visualizacoes(df_com_clusters, analise, generos):
    
    # Grafico 1: Rating medio por cluster
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    clusters_ordenados = analise.sort_values('cluster')
    cores = plt.cm.viridis(np.arange(len(clusters_ordenados)) / len(clusters_ordenados))
    plt.bar(clusters_ordenados['cluster'].astype(str), clusters_ordenados['rating_medio'], color=cores)
    plt.title('Rating Medio por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Rating Medio')
    plt.grid(True, alpha=0.3)
    
    # Grafico 2: Numero de jogos por cluster
    plt.subplot(1, 2, 2)
    contagem = df_com_clusters['cluster'].value_counts().sort_index()
    plt.bar(contagem.index.astype(str), contagem.values, color=plt.cm.Set3(range(len(contagem))))
    plt.title('Numero de Jogos por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Quantidade')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clusters_basicos.png', dpi=300)
    plt.show()
    
    # Grafico 3: Heatmap de generos
    if len(generos) > 0:
        plt.figure(figsize=(12, 6))
        
        # Preparar dados para heatmap
        dados_heatmap = []
        for cluster_id in sorted(df_com_clusters['cluster'].unique()):
            cluster_df = df_com_clusters[df_com_clusters['cluster'] == cluster_id]
            linha = []
            for genero in generos:
                if genero in cluster_df.columns:
                    pct = (cluster_df[genero].sum() / len(cluster_df)) * 100
                    linha.append(pct)
                else:
                    linha.append(0)
            dados_heatmap.append(linha)
        
        # Criar heatmap
        sns.heatmap(dados_heatmap, 
                   xticklabels=generos,
                   yticklabels=[f'Cluster {i}' for i in sorted(df_com_clusters['cluster'].unique())],
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.0f')
        
        plt.title('Percentagem de Generos por Cluster')
        plt.xlabel('Generos')
        plt.ylabel('Clusters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('heatmap_generos.png', dpi=300)
        plt.show()

# ============================================
# 7. RESPOSTA A PERGUNTA
# ============================================
def responder_pergunta(analise, df_com_clusters, generos):
    print("\n" + "="*50)
    print("RESPOSTA:")
    print("="*50)
    
    # Encontrar cluster com maior rating
    melhor_cluster = analise.iloc[0]
    
    print(f"\nPERGUNTA: Qual combinacao de generos tem os jogos com maior rating?")
    print(f"\nRESPOSTA: O Cluster {int(melhor_cluster['cluster'])} tem os melhores ratings.")
    
    print(f"\nCARACTERISTICAS DESTE CLUSTER:")
    print(f"-Rating medio: {melhor_cluster['rating_medio']:.2f}")
    print(f"-Numero de jogos: {int(melhor_cluster['n_jogos'])}")
    print(f"-Generos principais: {melhor_cluster['generos_principais']}")
    
    # Mostrar alguns exemplos
    cluster_exemplos = df_com_clusters[df_com_clusters['cluster'] == melhor_cluster['cluster']]
    
    print(f"\nEXEMPLOS DE JOGOS NESTE CLUSTER:")
    for _, jogo in cluster_exemplos.head(3).iterrows():
        # Encontrar generos do jogo
        generos_jogo = [g for g in generos if g in df_com_clusters.columns and jogo[g] == 1]
        print(f"-{jogo['name']} (Rating: {jogo['rating']:.1f})")
        if generos_jogo:
            print(f"    Generos: {', '.join(generos_jogo)}")
    
    return melhor_cluster

# ============================================
# 8. FUNCAO PRINCIPAL
# ============================================
def main():
    print("="*50)
    print("ANALISE DE CLUSTERING - JOGOS DE VIDEO")
    print("="*50)
    
    # 1. Carregar dados
    print("\n1. Carregando dados...")
    df, generos = carregar_dados()
    
    # 2. Preparar dados
    print("\n2. Preparando dados...")
    dados_cluster, scaler = preparar_dados(df, generos)
    
    # 3. Encontrar melhor K
    print("\n3. Encontrando melhor numero de clusters...")
    inercias = encontrar_melhor_k(dados_cluster)
    
    # 4. Aplicar K-means (usando K=6 por causa do metodo do cotovelo)
    print("\n4. Aplicando K-means clustering...")
    df_com_clusters, modelo_kmeans = aplicar_kmeans(df, dados_cluster, k=6)
    
    # 5. Analisar clusters
    print("\n5. Analisando clusters...")
    analise = analisar_clusters(df_com_clusters, generos)
    
    # 6. Criar visualizacoes
    print("\n6. Criando visualizacoes...")
    criar_visualizacoes(df_com_clusters, analise, generos)
    
    # 7. Responder pergunta
    print("\n7. Gerando resposta...")
    melhor_cluster = responder_pergunta(analise, df_com_clusters, generos)
    
    # 8. Salvar resultados
    print("\n8. Salvando resultados...")


    print("\n" + "="*50)
    print("ANALISE CONCLUIDA!")
    print("="*50)
    
    return df_com_clusters, analise

# ============================================
# EXECUTAR
# ============================================
if __name__ == "__main__":
    df_resultado, analise_resultado = main()