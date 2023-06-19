# --------------------------------------------
#
# EAD0830 - IA e ML Aplicados A Finan�as
# Prof. Leandro Maciel
# leandromaciel@usp.br
# Aula 7 - Regressao Logistica
#
# --------------------------------------------

# Carregar pacotes:
library(caret) # tecnicas classificao
library(readxl)
library(tidyverse) # funoeses para analise de dados
library(ROCR) 

# Carregar os dados:

setwd("C:/Users/Leandro/Desktop/Disciplinas FEA-USP/EAD 830 - IA e ML/Aula 7")
DadosAula7 <- read_excel("DadosAula7.xlsx",skip = 1) #skip - pular linha 1

# Caracter�sticas dos dados:

summary(DadosAula7)

# Frequ�ncia das classes:

count(DadosAula7,vars = Y)
count(DadosAula7,vars = Y)/nrow(DadosAula7) # frequencia relativa

# Determinar o tamanho da amostra treinamento:

per <- 0.75 # percentual da amostra toda para treinamento

n <- round(per*nrow(DadosAula7),0)

train <- DadosAula7[1:n,]
test <- DadosAula7[(n+1):nrow(DadosAula7),]

# Frequ�ncia das classes:

count(train,vars = Y)
count(test,vars = Y)
count(train,vars = Y)/nrow(train) # frequ�ncia relativa
count(test,vars = Y)/nrow(test) # frequ�ncia relativa

count(train,vars = X2) # vari�vel g�nero

# Estimar o modelo:

modeloRL <- glm(Y ~ X1 + X2 + X3 + X4 + X5 + X6,family=binomial(link='logit'),data=train)

# Sum�rio do modelo:

summary(modeloRL)

# Resultados para a base teste:

resTeste <- predict(modeloRL,newdata=test[,1:24],type='response')
resTeste <- ifelse(resTeste > 0.5,1,0)

# Confusion matrix and statistics
confusionMatrix(data=factor(resTeste), reference=factor(test$Y))

#No Information Rate -> melhor palpite quando nao ha informaoeses
#Assume classe de maior ocorr�ncia


# Seleoeso aleat�ria amostras treino e teste:

indices = sample(seq(1,nrow(DadosAula7),1),size = n,replace = FALSE)

train <- DadosAula7[c(indices),]

test <- DadosAula7[-c(indices),]
