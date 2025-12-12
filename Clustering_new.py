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
    #features = ['rating'] + [g for g in generos if g in df.columns]
    features =[g for g in generos if g in df.columns]
    
    # Criar dataset para clustering
    dados_cluster = df[features].copy()
    
    # Normalizar rating
    """ scaler = StandardScaler()
    dados_cluster['rating'] = scaler.fit_transform(dados_cluster[['rating']]) """
    
    # Garantir que generos sao inteiros
    for genero in generos:
        if genero in dados_cluster.columns:
            dados_cluster[genero] = dados_cluster[genero].astype(int)
    
    print(f"Features usadas: {len(features)}")
    #return dados_cluster, scaler
    return dados_cluster


# ============================================
# 3. ENCONTRAR MELHOR NUMERO DE CLUSTERS
# ============================================
def encontrar_melhor_k(dados):
    inercias = []
    silhuetas = []
    k_range = range(2, 11) 
    print("\nTestando diferentes numeros de clusters:")
    for k in k_range:
        # aplicar KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(dados)

        # calcular inércia
        inercias.append(kmeans.inertia_)

        # calcular silhueta
        silhueta = silhouette_score(dados, cluster_labels)
        silhuetas.append(silhueta)

        print(f"  K={k}: Inercia={kmeans.inertia_:.2f}, Silhueta={silhueta:.3f}")
    
 # Plot dos dois métodos lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico 1: Método do Cotovelo
    ax1.plot(k_range, inercias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inércia', fontsize=12)
    ax1.set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Silhouette Score
    ax2.plot(k_range, silhuetas, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Método Silhouette', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacao_cotovelo_silhueta.png', dpi=300, bbox_inches='tight')
    plt.show()
    return inercias, silhuetas 



# ============================================
# 4. APLICAR K-MEANS
# ============================================
def clustering(df, generos, k_escolhido=10):
  
    # Usar apenas generos
    X = df[generos].copy()
    
    print(f"Features usadas: {len(generos)} generos")
    print(f"Numero de clusters escolhido: K={k_escolhido}")
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=k_escolhido, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Calcular silhouette score
    silhueta = silhouette_score(X, df['cluster'])
    print(f"Silhouette Score: {silhueta:.3f}")
    
    # Interpretacao do score
    if silhueta > 0.7:
        print("EXCELENTE: Clusters muito bem definidos")
    elif silhueta > 0.5:
        print("BOM: Estrutura razoavel de clusters")
    elif silhueta > 0.25:
        print("FRACO: Alguma estrutura, mas sobreposicao")
    else:
        print("PESSIMO: Sem estrutura significativa")
     
 
    # Encontrar cluster com maior rating medio (se tiver rating)
    if 'rating' in df.columns:
        print(f"\nCLUSTER COM MELHOR RATING:")
        print("-" * 40)
        
        # Calcular rating medio por cluster
        cluster_ratings = []
        for cluster_id in range(k_escolhido):
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_ratings.append({
                'cluster': cluster_id,
                'rating_medio': cluster_df['rating'].mean(),
                'n_jogos': len(cluster_df)
            })
        
        # Ordenar do maior para o menor rating
        cluster_ratings.sort(key=lambda x: x['rating_medio'], reverse=True)
        
        # Pegar o melhor
        melhor = cluster_ratings[0]
        print(f"Cluster {melhor['cluster']}: Rating medio = {melhor['rating_medio']:.2f}")
        print(f"Tem {melhor['n_jogos']} jogos ({melhor['n_jogos']/len(df)*100:.1f}% do total)")
        
        # Analisar este cluster especifico
        cluster_melhor = df[df['cluster'] == melhor['cluster']]
        print(f"\nCaracteristicas do cluster {melhor['cluster']}:")
        
        # Encontrar generos mais comuns neste cluster
        generos_cluster = []
        for genero in generos:
            pct = cluster_melhor[genero].mean() * 100
            if pct > 30:  # Mais de 30%
                generos_cluster.append((genero, pct))
        
        # Ordenar generos
        generos_cluster.sort(key=lambda x: x[1], reverse=True)
        
        # Mostrar generos
        if generos_cluster:
            print(f"Generos predominantes: ", end="")
            for i, (genero, pct) in enumerate(generos_cluster[:5]):
                print(f"{genero} ({pct:.0f}%)", end=", " if i < len(generos_cluster[:5])-1 else "")
            print()
        else:
            print(f"Generos predominantes: Nenhum genero claro")
        
        # Mostrar exemplos de jogos
        print(f"\nExemplos de jogos neste cluster (maiores ratings):")
        top_jogos = cluster_melhor.nlargest(5, 'rating')[['name', 'rating']]
        
        for idx, jogo in top_jogos.iterrows():
            print(f"   {jogo['name']} (Rating: {jogo['rating']:.1f})")
    
    return df, kmeans
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
def responder_pergunta(df, generos, k=10):

    print(f"\nANALISE: Usando {k} clusters baseados apenas em generos")
    
    # Calcular silhouette score
    score = silhouette_score(df[generos], df['cluster'])
    print(f"Qualidade do clustering: {score:.3f}")
    
    if score > 0.5:
        print("Qualidade: BOA")
    elif score > 0.25:
        print("Qualidade: FRACA")
    else:
        print("Qualidade: PESSIMA")
    
    # Encontrar cluster com maior rating medio
    rating_por_cluster = df.groupby('cluster')['rating'].mean()
    melhor_cluster_id = rating_por_cluster.idxmax()
    melhor_rating = rating_por_cluster.max()
    
    print(f"\nCLUSTER COM MAIOR RATING MEDIO:")
    print(f"• Cluster numero: {melhor_cluster_id}")
    print(f"• Rating medio: {melhor_rating:.2f}")
    
    # Analisar este cluster
    cluster_df = df[df['cluster'] == melhor_cluster_id]
    
    print(f"\nINFORMACOES SOBRE ESTE CLUSTER:")
    print(f"• Total de jogos: {len(cluster_df)}")
    
    print(f"\nGENEROS PRINCIPAIS (mais de 50% dos jogos tem):")
    generos_principais = []
    
    for genero in generos:
        pct = cluster_df[genero].mean() * 100
        if pct > 50:
            generos_principais.append(genero)
            print(f"  • {genero}: {pct:.0f}%")
    
    
    # Encontrar combinacao mais comum
    print(f"\nCOMBINACAO DE GENEROS MAIS COMUM:")
    
    # Criar string com combinacao de generos para cada jogo
    combinacoes = []
    for idx, jogo in cluster_df.iterrows():
        generos_jogo = [g for g in generos if jogo[g] == 1]
        combinacao_str = "+".join(sorted(generos_jogo)) if generos_jogo else "Nenhum"
        combinacoes.append(combinacao_str)
    
    # Contar frequencia de cada combinacao
    from collections import Counter
    contagem = Counter(combinacoes)
    
    # Pegar combinacao mais comum
    combinacao_mais_comum, frequencia = contagem.most_common(1)[0]
    
    print(f"Combinacao: {combinacao_mais_comum}")
    print(f"Quantidade: {frequencia} jogos")
    print(f"Porcentagem: {frequencia/len(cluster_df)*100:.1f}% dos jogos do cluster")
    
    # Mostrar exemplos de jogos
    print(f"\nEXEMPLOS DE JOGOS DESTE CLUSTER:")
    
    # Pegar 5 jogos com maior rating
    top_jogos = cluster_df.nlargest(5, 'rating')
    
    for i, (idx, jogo) in enumerate(top_jogos.iterrows(), 1):
        # Encontrar generos deste jogo
        generos_jogo = [g for g in generos if jogo[g] == 1]
        
        print(f"\n{i}. {jogo['name']}")
        print(f"   Rating: {jogo['rating']:.1f}")
        print(f"   Generos: {', '.join(generos_jogo) if generos_jogo else 'Nenhum'}")
    
    # Comparar com outros clusters
    print(f"\nCOMPARACAO COM OUTROS CLUSTERS (top 3 por rating):")
    print("-" * 50)
    
    # Pegar 3 melhores clusters por rating
    top_3_clusters = rating_por_cluster.nlargest(3)
    
    for posicao, (cluster_id, rating_medio) in enumerate(top_3_clusters.items(), 1):
        cluster_temp = df[df['cluster'] == cluster_id]
        
        # Encontrar generos principais deste cluster
        generos_cluster = []
        for genero in generos:
            pct = cluster_temp[genero].mean() * 100
            if pct > 50:
                generos_cluster.append(genero)
        
        print(f"{posicao}. Cluster {cluster_id}:")
        print(f"   Rating medio: {rating_medio:.2f}")
        print(f"   Jogos: {len(cluster_temp)}")
        print(f"   Generos principais: {', '.join(generos_cluster) if generos_cluster else 'Variados'}")
        print()
    
    # Resumo final
    print(f"\nRESUMO FINAL:")
    print("-" * 50)
    
    if generos_principais:
        print(f"Os jogos com maior rating tendem a ter estes generos:")
        for genero in generos_principais:
            print(f"{genero}")
        
        print(f"\nEsta combinacao aparece em {len(cluster_df)} jogos")
        print(f"com rating medio de {melhor_rating:.2f}")
        
        if combinacao_mais_comum != "Nenhum":
            print(f"A combinacao especifica mais comum e: {combinacao_mais_comum}")
    else:
        print(f"Nao ha uma combinacao clara de generos para jogos com alto rating.")
        print(f"Os jogos com maior rating sao diversos em termos de generos.")
    
    return melhor_cluster_id, cluster_df



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
    dados_cluster = preparar_dados(df, generos)
    
    # 3. Encontrar melhor K
    print("\n3. Encontrando melhor numero de clusters...")
    inercias, silhuetas = encontrar_melhor_k(dados_cluster)
    
    # 4. Aplicar K-means (usando K=6 por causa do metodo do cotovelo)
    print("\n4. Aplicando K-means clustering...")
    df_clustered, kmeans_model = clustering(df, generos, k_escolhido=10)
    
    # 5. Analisar clusters
    print("\n5. Analisando clusters...")
    analise = analisar_clusters(df_clustered, generos)
    
    # 6. Criar visualizacoes
    print("\n6. Criando visualizacoes...")
    criar_visualizacoes(df_clustered, analise, generos) 
    
    # 7. Responder pergunta
    print("\n7. Gerando resposta...")
    responder_pergunta(df_clustered, generos, k=10)
    
    # 8. Salvar resultados
    print("\n8. Salvando resultados...")


    print("\n" + "="*50)
    print("ANALISE CONCLUIDA!")
    print("="*50)
    
    return df_clustered, analise

# ============================================
# EXECUTAR
# ============================================
if __name__ == "__main__":
    df_resultado, analise_resultado = main()