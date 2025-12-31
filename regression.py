import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import EDA_new as dp


def prepare_features(df_clean):

    print("\n--- PREPARANDO DADOS PARA REGRESSÃO ---")
    df_model = df_clean.copy()

    #remove coluna desnecessarias
    if 'name' in df_model.columns:
        df_model = df_model.drop(columns=['name'])
    if 'plot' in df_model.columns:
        df_model = df_model.drop(columns=['plot'])    
    if 'url' in df_model.columns:
        df_model = df_model.drop(columns=['url'])
    if 'game_id' in df_model.columns:
        df_model = df_model.drop(columns=['game_id'])

    # Transformar 'certificate' em colunas numéricas (0 ou 1)
    if 'certificate' in df_model.columns:
   #     print("Realizando One-Hot Encoding na coluna 'certificate'...")
        df_model = pd.get_dummies(df_model, columns=['certificate'], drop_first=True)
        
    # Garantir que tudo é numérico
    print(f"Colunas finais para o modelo ({df_model.shape[1]}): {df_model.columns.tolist()}")
    return df_model

def run_linear_regression(df_model):
    
    #dividir base
    target = 'rating'
    X = df_model.drop(columns=[target]) #teste 
    y = df_model[target]                #treino
    
    # Divisão (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDivisão (80/20)")
    print(f"Treino: {X_train.shape[0]} jogos")
    print(f"Teste:  {X_test.shape[0]} jogos")

    # Criar e Treinar o Modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # previsao
    y_pred = model.predict(X_test)
    
    # Avaliação (Erro Médio)
    r2 = r2_score(y_test, y_pred)
    mean = mean_absolute_error(y_test, y_pred)
    
    print(f"\nRESULTADOS DA REGRESSÇAO LINEAR:")
    print(f"Score R²: {r2:.4f}")
    print(f"Erro Médio: {mean:.4f}")
    print(f"O modelo explica {r2*100:.2f}% da variação nas notas dos jogos.")

    
    return model, X_train, y_test, y_pred

def predict(model,X_train):
    print("\nFazer Previsão de rating")

    generos = [c for c in X_train.columns if c not in ['year', 'votes'] and not c.startswith('certificate')]
    print(f"Gêneros disponiveis: {generos}")
    genero_input= input("Gênero: ")

    coluna_encontrada = None
    for col in X_train.columns:
        if col.lower() == genero_input.lower():
            coluna_encontrada = col
            break

    if not coluna_encontrada:
        print(f"Erro: Gênero '{genero_input}' não encontrado.")
        return
    

    year_input = int(input("Digite o Ano de Lançamento: "))
    input_data = pd.DataFrame(0, index=[0], columns=X_train.columns)
 
    input_data[coluna_encontrada] = 1
    input_data['year'] = year_input
    input_data['votes'] = X_train['votes'].median() #usamos a mediana nos votos para nao penalizar a nota
    
    # previsao
    prediction = model.predict(input_data)[0]
    
    print("\nRESULTADO DA SIMULAÇÃO")
    print(f"Jogo: Gênero {coluna_encontrada}, Ano {year_input}")
    print(f"NOTA PREVISTA: {prediction:.2f} / 10.0")  

    return


def analyze(model, feature_names):
    print("\nANÁLISE")
    
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    # Ordenar por impacto absoluto
    coefs['Coefficient'] = coefs['Coefficient'].round(2)
    coefs = coefs.sort_values(by='Coefficient', ascending=False)

    print("\nTop 5 Fatores que aumentam a nota:")
    print(coefs.head(5).to_string(index=False))
    
    print("\nTop 5 Fatores que diminuem a nota:")
    print(coefs.tail(5).to_string(index=False))


def main_modeling():
    #importar EDA
    df_clean = dp.main() 
    
    df_model = prepare_features(df_clean)
    model, X_train, y_test, y_pred = run_linear_regression(df_model)
    analyze(model,X_train.columns) 
    predict(model, X_train)
    

if __name__ == "__main__":
    main_modeling()