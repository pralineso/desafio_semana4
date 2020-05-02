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

# In[95]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[44]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[45]:


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

# In[5]:


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

# In[6]:


def q2():
    # Retorne aqui o resultado da questão 2.
    #intervalo = [15.977, 24.005]
    #P(15 <= x <= 24) == F(24) - F(15) == F(intervalo)
    x = round(np.mean(dataframe.normal))
    s = round(np.std(dataframe.normal))
    #[𝑥¯−𝑠,𝑥¯+𝑠] 
    return float(round(round(sct.norm.cdf(x+s, loc=x, scale=s), 3)-round(sct.norm.cdf(x-s, loc=x, scale=s), 3),3))
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

# In[7]:


def q3():
    # Retorne aqui o resultado da questão 3.
    m_binom = round(np.mean(dataframe.binomial), 3)
    m_norm = round(np.mean(dataframe.normal), 3)
    v_binom = round(np.var(dataframe.binomial), 3)
    v_norm = round(np.var(dataframe.normal), 3)
    return (round(m_binom-m_norm, 3), round(v_binom-v_norm, 3))
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude? *nao, pelo menos nao pra variancia*
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`? *nao entedi essa ainda*

# ## Parte 2

# ### _Setup_ da parte 2

# In[3]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[85]:


stars.shape


# In[71]:


# Sua análise da parte 2 começa aqui.

#coluna filtrada
star_mean_profile = pd.DataFrame(stars['mean_profile'][stars['target'] == 0])


# In[86]:


star_mean_profile.shape


# In[72]:


#salvando a media o desvio padrao
#x = media
#s = desvio padrao
x = star_mean_profile.mean()
s = star_mean_profile.std()
print(x,"\n", s)


# In[73]:


false_pulsar_mean_profile_standardized = sct.zscore(star_mean_profile)


# In[80]:


#test.mean()
#test.var()
mean_profile_standardized = pd.DataFrame(false_pulsar_mean_profile_standardized)


# In[81]:


mean_profile_standardized


# In[103]:


round(mean_profile_standardized.var())


# In[109]:


round(mean_profile_standardized.std())


# In[108]:


round(mean_profile_standardized.mean())


# In[ ]:


#print(dataframe.normal.quantile(q=0.25))


# In[96]:


from scipy.stats import norm


# In[135]:


vals = norm.ppf([0.80, 0.90, 0.95])
print(vals)


q1 = round(sct.norm.ppf(0.80, loc=0, scale=1), 3) # loc = media, scale = desvio padrão
q2 = round(sct.norm.ppf(0.90, loc=0, scale=1), 3)
q3 = round(sct.norm.ppf(0.95, loc=0, scale=1), 3)

#print(q1,q2,q3)

#print(sct.norm.ppf(q1, loc=0, scale=1))
#print(sct.norm.ppf(q2, loc=0, scale=1))
#print(sct.norm.ppf(q3, loc=0, scale=1))
#print(sct.norm.isf(q1, loc=0, scale=1))

[round(sct.norm.isf(q1, loc=0, scale=1), 3), round(sct.norm.isf(q2, loc=0, scale=1), 3), round(sct.norm.isf(q3, loc=0, scale=1), 3)]


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

# In[10]:


def q4():
    # Retorne aqui o resultado da questão 4.
    q1 = round(sct.norm.ppf(0.80, loc=0, scale=1), 3) # loc = media, scale = desvio padrão
    q2 = round(sct.norm.ppf(0.90, loc=0, scale=1), 3)
    q3 = round(sct.norm.ppf(0.95, loc=0, scale=1), 3)

    return [round(sct.norm.isf(q1, loc=0, scale=1), 3), round(sct.norm.isf(q2, loc=0, scale=1), 3), round(sct.norm.isf(q3, loc=0, scale=1), 3)]
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
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
