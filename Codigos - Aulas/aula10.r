

# ------------------------------------------------
# FEAUSP
# EAD830 - IA e ML Aplicados a Financas
# Prof. Leandro Maciel (leandromaciel@usp.br)
# 
# ------------------------------------------------


# Carregar os pacotes necessarios:

library(plot3D)
library(GA)
library(parallel)
library(doParallel)
library(readxl)


# ------------------------------------------------

cat("\f") # Limpar o console

rm(list = ls()) # Limpar todas as variaveis

# ------------------------------------------------

# Carregar os dados dos retornos das acoes:

setwd("C:/Users/Leandro/Desktop/Disciplinas FEA-USP/EAD 830 - IA e ML/Aula 9")

Acoes = read_excel("Acoes.xlsx",sheet = "Retornos")

# ------------------------------------------------

noAcoes = ncol(Acoes) - 1 # numero de acoes (primeira coluna datas)

# Dividir amostra antes e depois da formacao de carteiras...

# Antes, periodo de 3/1/12 a 29/12/2016:

Retornos = Acoes[1:1236,2:ncol(Acoes)] # divisao com base nos dados
RetornosFora = Acoes[1237:1935,2:ncol(Acoes)]

# ------------------------------------------------

# Fun��o calcula retorno da carteira:
# Tem como o argumento o vetor de pesos...

portfolio_returns = function(x){
  port.returns = 0
  
  # Multiplicamos o retorno do ativo pelo peso:
  for (i in 1:length(x)) {
    port.returns = port.returns + Retornos[,i]*x[i]
  }
  return (port.returns)
}

# Fun��o que calcula o risco da carteira:
# Tem como o argumento o vetor de pesos...

cvm = function(x) {
  port.returns = portfolio_returns(x) # calcula os retornos...
  return(sd(port.returns)) # risco como o desvio padr�o
}

# Restri��o soma pesos igual a 1:

constraint = function(x) {
  boundary_constr = (sum(x)-1)^2 # Carteira totalmente investida
  return(boundary_constr)
}

# Fun��o objetivo:

obj = function(x) {
  return(cvm(x)+100*constraint(x))
}

# Rodar o AG:

ga_res = ga(
  type = "real-valued", # solu��o s�o n�meros reais
  function(x){-obj(x)}, # fun��o objetivo (default maximiza��o)
  lower = rep(0,ncol(Retornos)), # limite m�nimo pesos
  upper = rep(1,ncol(Retornos)), # limite m�ximo pesos 
  popSize = 60, # default - tamanho da popula��o
  pcrossover = 0.8, # default - taxa de crossover
  pmutation = 0.1, # default - taxa de muta��o
  elitism = max(1, round(50*0.05)), # default - 5% do popSize
  maxiter = 50000, # n�mero m�ximo itera��es 
  run = 50, # se fun��o objetivo n�o mudar em 50 itera��es, parar 
  parallel = TRUE, # processamento em paralelo (mais r�pido)
  monitor = TRUE, # aparecer evolu��o das opera��es
  seed = 1 # semente para replicarmos resultados (gera��o pop inicial)
)

# Apresentar resultados principais:

summary(ga_res)

# Guardar os pesos �timos em um vetor:

sol = as.vector(summary(ga_res)$solution)

# Fun��o para calcular o retorno da carteira fora da amostra:

portfolio_returns_test = function(x) {
  port.returns = 0
  for (i in 1:length(x)) {
    port.returns = port.returns + RetornosFora[,i] * x[i]
  }
  return (port.returns)
}

# Calcular os retornos na amostra teste com a carteira escolhida:

retornos_teste = portfolio_returns_test(sol)

# retorno m�dio:

mean(retornos_teste[,1])*100

# resultado aula 9 --> 0.0928

# Calcular o retorno acumulado ao final do per�odo:

cumsum(retornos_teste[,1])[nrow(retornos_teste)]*100 # em percentual

# resultado aula 9 --> 0.83

# Calcular o risco na amostra teste (desvio-padr�o):

sd(retornos_teste[,1])*100 # em percentual

# resultado aula 9 --> 1.052612

# Plotar os retornos da carteira na amostra teste:

plot(cumsum(retornos_teste[,1]),type="l",lwd=2,ylim=c(-0.1,0.85))




#-----------------------------------------------------