# --------------------------------------------
#
# EAD0830 - IA e ML Aplicados A Financas
# Prof. Leandro Maciel
# leandromaciel@usp.br
# Aula 8 - K-NN
#
# --------------------------------------------

# Carregar pacotes:
library(caret) # tecnicas classificaaoo
library(readxl)
library(tidyverse) # funaoes para analise de dados
library(ROCR) 
library(class)

# Carregar os dados:

setwd("C:/Users/Leandro/Desktop/Disciplinas FEA-USP/EAD 830 - IA e ML/Aula 7")
DadosAula7 <- read_excel("DadosAula7.xlsx",skip = 1) #skip - pular linha 1

# Determinar o tamanho da amostra treinamento:

per <- 0.75 # percentual da amostra toda para treinamento

n <- round(per*nrow(DadosAula7),0)

# Criar a funaoo para normalizar:

normalize = function(x){
  
  return ((x-min(x))/(max(x)-min(x)))
  
}

# Normalizar os dados:

DataNorm = sapply(DadosAula7,normalize)

# Dividir amostras:

train <- DataNorm[1:n,]
test <- DataNorm[(n+1):nrow(DataNorm),]

# Definir o valor de k:

k = round(sqrt(nrow(train)),0) # definir os k vizinhos

# Classificar na amostra teste:

modelo_kNN = knn(train[,2:7],test[,2:7],train[,25],k)

# Matriz de confusao:

confusionMatrix(data=factor(modelo_kNN), reference=factor(test[,25]))

# Regressao Logistica:

train <- DadosAula7[1:n,]
test <- DadosAula7[(n+1):nrow(DadosAula7),]
modeloRL <- glm(Y ~ X1 + X2 + X3 + X4 + X5 + X6,family=binomial(link='logit'),data=train)
resTeste <- predict(modeloRL,newdata=test[,1:24],type='response')
resTeste <- ifelse(resTeste > 0.5,1,0)

# Matriz de confus�o:
confusionMatrix(data=factor(resTeste), reference=factor(test$Y))
