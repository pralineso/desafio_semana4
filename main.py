#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[84]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[85]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[5]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[ ]:





# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[127]:


def q1():
    # Retorne aqui o resultado da questão 1.
    q1_norm = np.percentile(dataframe.normal, 25)
    q2_norm = np.percentile(dataframe.normal, 50)
    q3_norm = np.percentile(dataframe.normal, 75)
    q1_binom = np.percentile(dataframe.binomial, 25)
    q2_binom = np.percentile(dataframe.binomial, 50)
    q3_binom = np.percentile(dataframe.binomial, 75)
    return (round(q1_norm - q1_binom, 3), round(q2_norm - q2_binom, 3), round(q3_norm - q3_binom, 3))
    pass


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# *sim, pelos valores listados no describe*
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?
# *ainda nao*

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[112]:


def q2():
    # Retorne aqui o resultado da questão 2.
    #intervalo = [15.977, 24.005]
    #P(15 <= x <= 24) == F(24) - F(15) == F(intervalo)
    x = np.mean(dataframe.normal)
    s = np.std(dataframe.normal)
    #print(x,s)
    #[𝑥¯−𝑠,𝑥¯+𝑠]   
    b = x+s
    a = x-s
    Fb = sct.norm.cdf(b, loc=20, scale=4)
    Fa = sct.norm.cdf(a, loc=20, scale=4)
    return float(round(Fb-Fa,3))
    pass


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico? *acho q nao, achei muio estranho os valores, se fiz certo achei o desvio padrao mto alto*
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$. *nao sei dizer se melhorou ou piorou, so sei q mdeu um resultado maior, masi porximo de 1*

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[114]:


def q3():
    # Retorne aqui o resultado da questão 3.
    m_binom = np.mean(dataframe.binomial)
    m_norm = np.mean(dataframe.normal)
    v_binom = np.var(dataframe.binomial)
    v_norm = np.var(dataframe.normal)
    return (round(m_binom-m_norm, 3), round(v_binom-v_norm, 3))
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude? *nao, pelo menos nao pra variancia*
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`? *nao entedi essa ainda*

# ## Parte 2

# ### _Setup_ da parte 2

# In[4]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[34]:


# Sua análise da parte 2 começa aqui.


# In[5]:


#coluna filtrada
#false_mean_profile = pd.DataFrame(stars['mean_profile'][stars['target'] == 0])


# In[35]:


#padronizando
#false_mean_profile.describe()
#false_pulsar_mean_profile_standardized = false_mean_profile.apply(lambda x: ((x-116.561250)/17.475456))
#os valores ali na formula sao os valores da media e do desvio padrao que aparece quando da o .describe()
#false_pulsar_mean_profile_standardized


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[118]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return (0,0,0)
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[11]:


def q5():
    # Retorne aqui o resultado da questão 5.
    #nao sei pq mas so da certo se isso tiver aqui 
    false_mean_profile = pd.DataFrame(stars['mean_profile'][stars['target'] == 0])
    false_pulsar_mean_profile_standardized = false_mean_profile.apply(lambda x: ((x-116.561250)/17.475456))
    Q1_df = np.quantile(false_pulsar_mean_profile_standardized, .25) 
    Q2_df = np.quantile(false_pulsar_mean_profile_standardized, .50)
    Q3_df = np.quantile(false_pulsar_mean_profile_standardized, .75)
    Q_teoricos2 = sct.norm.ppf([0.25, 0.50, 0.75])
    return (round(Q1_df-Q_teoricos2[0],3), round(Q2_df-Q_teoricos2[1],3), round(Q3_df-Q_teoricos2[2],3))
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
