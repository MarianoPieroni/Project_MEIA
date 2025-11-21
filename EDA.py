import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df =pd.read_csv("imdb-videogames.csv")
df_clean = df.copy()

print("DataSet")
""" print(f"\ndimensoes{df.shape}")
print(f"\ncolunas\n{df.columns}")
print(f"\nvalores nulos\n{df.isnull().sum()}")
print(f"total de nulos\n{df.isnull().sum().sum()}")
print(f"\nvalores duplicados\n{df.duplicated().sum()}")
print(f"\ntipos de dados\n{df.dtypes}") """

print(df_clean.describe(include='all'))

#CLASSIFICAÇÃO DAS VARIÁVEIS
variables = {
    'Numericas': ['year', 'rating', 'votes'],
    'Categoricas': ['name', 'url', 'certificate', 'plot'],
    'Binarias': ['Action', 'Adventure', 'Comedy', 'Crime', 'Family', 
                'Fantasy', 'Mystery', 'Sci-Fi', 'Thriller']
}

print("Variavel Dependente (target): 'rating'")
print("Variaveis Independentes: 'year', 'votes', 'certificate', gêneros")
print(f"valores duplicaldos:{df_clean.duplicated().sum()}")
print("\nVALORES Nulos:")
missing_data = df_clean.isnull().sum()
missing_data_total = df_clean.isnull().sum().sum()
print(missing_data[missing_data > 0])
print(f"Total de valores nulos: {missing_data_total}")
print(f"Total de linhas no dataset: {df_clean.shape[0]}")

# LIMPEZA DOS DADOS

#TRATAMENTO DE VALORES NULOS
#votes->substitiuir por zero 
df_clean['votes'] = df_clean['votes'].str.replace(',', '').astype(float)
df_clean['votes'] = df_clean['votes'].fillna(0)

#rating->substituir pela mediana
df_clean['rating']=df_clean['rating'].fillna(df_clean['rating'].median())

#year - Manter como nulo ou usar moda
# Se poucos nulos: eliminar essas linhas
if df_clean['year'].isna().sum() < len(df_clean) * 0.05:  # menos de 5%
        df_clean = df_clean.dropna(subset=['year'])
else:
        # Se muitos: substituir pela moda 
        df_clean['year'] = df_clean['year'].fillna(df_clean['year'].mode()[0])

df_clean['certificate'] = df_clean['certificate'].fillna('Not Rated')

print("\nDados apos limpeza:")
print("VALORES Nulos:")
missing_data = df_clean.isnull().sum()
missing_data_total = df_clean.isnull().sum().sum()
print(missing_data[missing_data > 0])
print(f"Total de valores nulos: {missing_data_total}")
print(f"Total de linhas no dataset: {df_clean.shape[0]}")

#so copiei e colei do gpt o boxplot
print("VISUALIZACAO DE OUTLIERS (BOXPLOTS):")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(data=df_clean, y='rating', ax=axes[0])
axes[0].set_title('Outliers - Rating')

sns.boxplot(data=df_clean, y='votes', ax=axes[1])
axes[1].set_title('Outliers - Votes')

sns.boxplot(data=df_clean, y='year', ax=axes[2])
axes[2].set_title('Outliers - Year')

plt.tight_layout()
plt.show()

# Método IQR para detectar outliers
def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers

print("\nOUTLIERS DETECTADOS:")
for col in ['rating', 'votes', 'year']:
    outliers = detect_outliers_iqr(df_clean[col])
    print(f"{col}: {len(outliers)} outliers")

# Normalização Min-Max(FICA COMENTADO POR AGORA PORQUE E CAPAZ DE NAO SE USAR)
""" print("\n" + "="*50)
print("NoRMALIZAÇÃO")
scaler = MinMaxScaler()
df_clean[['rating_norm', 'votes_norm']] = scaler.fit_transform(df_clean[['rating', 'votes']]) """

# Codificação de variáveis
""" certificate_dummies = pd.get_dummies(df_clean['certificate'], prefix='cert')
df_clean = pd.concat([df_clean, certificate_dummies], axis=1)

print("Variáveis categóricas codificadas")

# Variáveis derivadas
df_clean['engagement_score'] = df_clean['rating'] * np.log1p(df_clean['votes'])
df_clean['is_modern'] = (df_clean['year'] > 2000).astype(int) #nao sei se e vamos usar

print(df_clean[['rating', 'rating_norm', 'votes', 'votes_norm']].head()) """

