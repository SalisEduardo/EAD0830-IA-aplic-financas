
# ------------------------------------------------
# FEAUSP
# EAD830 - IA e ML Aplicados a Finan�as
# Prof. Leandro Maciel (leandromaciel@usp.br)
# ------------------------------------------------

# Script Aula 6 - Redes Neurais Artificiais

# ------------------------------------------------

# Carregar os pacotes necessarios:

library(readxl)
library(neuralnet)

# ------------------------------------------------

cat("\f") # Limpar o console

rm(list = ls()) # Limpar todas as vari�veis

# ------------------------------------------------

# Carregar os dados IPCA (variacao % mensal Jan/00 a Jun/22):

setwd("C:/Users/Leandro/Desktop/Disciplinas FEA-USP/EAD 830 - IA e ML/Aula 5")
IPCA <- read_excel("DadosAula5.xls")

# Plotar os dados:

plot(IPCA$IPCA,type="l",ylab = "IPCA",xlab = "")
summary(IPCA) # Estat�sticas descritivas


# ------------------------------------------------

# Passo 1: normalizacao dos dados

# Construir fun��o para normalizar (min-max):

x_min=min(IPCA$IPCA)
x_max=max(IPCA$IPCA)

normalize = function(x){
  return((x-x_min)/(x_max-x_min))
}

# Normalizar os dados:

IPCA_norm = normalize(IPCA$IPCA)

plot(IPCA_norm,type = "l",xlab = "",ylab = "IPCA norm",col = "blue")

# ------------------------------------------------

# Passo 2: definir modelo de previsao...

# Quantos valores passados para prever o IPCA do proximo mes?

no = length(IPCA_norm)

base = as.data.frame(cbind(IPCA_norm[1:(no-2)],IPCA_norm[2:(no-1)],IPCA_norm[3:no]))

# Ver Planilha 2 dos dados

# ------------------------------------------------

# Passo 3: dividir amostras em treinamento e validacao:

n = 24 # numero de obs deixadas para previs�o (mesmo ARIMA)

IPCAin = base[1:(nrow(base)-n),]
IPCAout = base[(nrow(base)-n+1):(nrow(base)),]

# ------------------------------------------------

# Passo 4: definir estrutura e treinar RNA:

repeticoes = 5
modeloRede = neuralnet(
              V3 ~ V1 + V2, # modelo de previsao considerado;
              data = IPCAin, # base de dados treinamento;
              act.fct = "tanh", # tangente hiperbolica;
              hidden = c(2,2), # camadas e neuronios em cada uma;
              rep = repeticoes # n�mero de repeti��es (problema inicializa��o).
              )

# Visualizar rede neural:

plot(modeloRede,rep="best")

# Guardar a saida do modelo no treinamento:

saida_Treino = as.matrix(modeloRede[["net.result"]][[1]])

# Visualizar ajuste da rede no treinamento:

plot(IPCAin[,3],xlab = "",ylab = "Variacao (%)",type="l",main = "IPCA Real e Previsto - Amostra Treino")
lines(saida_Treino,col="red")

# ------------------------------------------------

# Passo 5: realizar previs�es 1 passo na amostra teste.

previsao = predict(modeloRede,IPCAout[,1:2],rep = repeticoes)

# Visualizacao:

plot(IPCAout[,3],xlab = "",ylab = "Variacao (%)",type="l",main = "IPCA Real e Previsto - Amostra Teste")
lines(previsao,col = "red")


# ------------------------------------------------

# Calcular o RMSE (dados normalizados)

desnormalize = function(x){
  return((x*(x_max-x_min) + x_min))
}

previsaoDes = desnormalize(previsao)
real = desnormalize(IPCAout[,3])
#plot(real,type="l")
#lines(previsaoDes,type="l",col="red")

# RMSE

rmse = sqrt( sum((real - previsaoDes)^2)*(1/n))
rmse

#0.3950658 RMSE ARIMA

# ------------------------------------------------
