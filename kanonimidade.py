import pandas as pd
import numpy as np
from pycanon import anonymity 
import random

# 20 20 21 23 25 33 35 36 44
# 2020 212325  
def agrupar_e_anonimizar(dados, k, coluna_faixa='Idade', coluna_cep='CEP'):
    df = dados.copy()
    df_anonimizado = pd.DataFrame()
    suprimidos = []

    # Agrupar por faixa de idade + prefixo de CEP (ex: 5***, 6***)
    grupos = df.groupby([coluna_faixa, coluna_cep])

    for (faixa, cep), grupo in grupos:
        grupo = grupo.reset_index(drop=True)
        tamanho = len(grupo)

        if tamanho < k:
            # Suprime completamente se o grupo for menor que k
            grupo_suprimido = grupo.applymap(lambda _: '**')
            suprimidos.append(grupo_suprimido)
            continue

        # Divide em subgrupos de tamanho k
        subgrupos = [grupo[i:i+k].copy() for i in range(0, tamanho, k)]
        
        if len(subgrupos) > 1 and len(subgrupos[-1]) < k:
            # Junta com o anterior se for da mesma faixa
            if subgrupos[-1][coluna_faixa].iloc[0] == subgrupos[-2][coluna_faixa].iloc[0]:
                subgrupos[-2] = pd.concat([subgrupos[-2], subgrupos[-1]])
                subgrupos.pop()  # Remove o último
            else:
                grupo_suprimido = subgrupos[-1].applymap(lambda _: '**')
                suprimidos.append(grupo_suprimido)
                subgrupos.pop()  # Remove o último original

        for subgrupo in subgrupos:
            if len(subgrupo) < k:
                # Como já tratamos antes, isso aqui é apenas segurança
                grupo_suprimido = subgrupo.applymap(lambda _: '**')
                suprimidos.append(grupo_suprimido)
                continue
            
            # >>>>> Padronizar Estado_Civil <<<<<
            modo_estado_civil = subgrupo['Estado_Civil'].mode()
            if not modo_estado_civil.empty:
                estado_padrao = modo_estado_civil.iloc[0]  # Pega o mais frequente (ou qualquer um em caso de empate)
                subgrupo.loc[:, 'Estado_Civil'] = estado_padrao

            df_anonimizado = pd.concat([df_anonimizado, subgrupo])

    # Adiciona os suprimidos ao final
    if suprimidos:
        df_suprimido = pd.concat(suprimidos, ignore_index=True)
        df_anonimizado = pd.concat([df_anonimizado, df_suprimido], ignore_index=True)

    return df_anonimizado.reset_index(drop=True)


def mascarar_cep(cep):
    cep_str = ''.join(filter(str.isdigit, str(cep)))
    resposta = cep_str[:3] + '***'
    return resposta

def valor_esta_no_intervalo(valor, intervalo_str):
    try:
        minimo, maximo = intervalo_str.replace('–', '-').split('-')
        minimo = int(minimo)
        maximo = int(maximo)
        return minimo <= int(valor) <= maximo
    except:
        return False

def comparar_cep(orig_cep, anon_cep):
    # Compara se os 3 primeiros dígitos do CEP original batem com o prefixo anon_cep (ex: "500***")
    return str(orig_cep).startswith(str(anon_cep).replace('*', '')[:3])

def linkage_attack_completo(original, anonimizado, quasi_ids):
    colunas_comuns = ['Idade', 'CEP', 'Estado_Civil']
    print(f"Colunas que serão comparadas: {colunas_comuns}")

    reidentificados = []

    for i, row_orig in original.iterrows():
        condicoes = pd.Series([True] * len(anonimizado))

        for col in colunas_comuns:
            val_orig = row_orig[col]
            col_anon = anonimizado[col]

            if col in quasi_ids:
                if col == 'Idade':
                    # Intervalos como "20–25" ou "10000-20000"
                    cond_col = col_anon.apply(lambda x: valor_esta_no_intervalo(val_orig, x) if not pd.isna(x) and not x.startswith("**") else False)
                elif col == 'CEP':
                    # Comparação por prefixo (ex: 500***)
                    cond_col = col_anon.apply(lambda x: comparar_cep(val_orig, x) if not pd.isna(x) and not x.startswith("***") else False)
                else:
                    # Quasi-id sem suprimir: igualdade exata, ignorando "**"
                    cond_col = col_anon.apply(lambda x: x == val_orig if not pd.isna(x) and x != "**" else False)
            else:
                # Colunas não QI: só compara se não estão suprimidas
                cond_col = col_anon.apply(lambda x: x == val_orig if not pd.isna(x) and x != "**" else False)

            condicoes &= cond_col

        candidatos = anonimizado[condicoes]
        if not candidatos.empty:
            reidentificados.append({
                **row_orig.to_dict(),
                'matches_no_anonimizado': len(candidatos)
            })

    reid_df = pd.DataFrame(reidentificados)
    reid_df.to_csv('reid_df.csv', index=False)
    taxa = len(reid_df) / len(original) if len(original) > 0 else 0

    print(f"\nEstatísticas:")
    print(f"Registros originais: {len(original)}")
    print(f"Registros reidentificados: {len(reid_df)}")
    print(f"Taxa de reidentificação: {taxa:.2%}")

    if not reid_df.empty:
        print("\nExemplos de registros reidentificados:")
        print(reid_df.head())
    else:
        print("\nNenhum registro reidentificado")

    return reid_df, taxa

# Exemplo de uso
dados = pd.read_csv('dataset_privado.csv')

#Suprimir informações pessoais
dados['Nome'] = '**'
dados['Sexo'] = '**'
dados['Escolaridade'] = '**'

bins = [0, 20, 26, 32, 39, 45, 61, 70, 100]
labels = ['0–19', '20–25', '26–31', '32–38', '39-45', '46-60', '61-69', '70+']
dados['Idade'] = pd.cut(dados['Idade'].astype(int), bins=bins, labels=labels, right=False)

dados['CEP'] = dados['CEP'].astype(str)
dados['CEP'] = dados['CEP'].apply(mascarar_cep)
quasi_ids = ['Idade', 'CEP', 'Estado_Civil']  # Colunas que serão anonimizadas como intervalos

#VALORES TESTES DE K
#k = 5
k = 10

dados_anon = agrupar_e_anonimizar(dados, k)
dados_anon.to_csv('dados_anonimizados.csv', index=False)

dados_publicos = pd.read_csv('dataset_publico.csv')
linkage_attack_completo(dados_publicos, dados_anon, quasi_ids)