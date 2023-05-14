
#-----------------------------------------------------
#
# Modelos ARIMA
# Prof. Leandro Maciel
# e-mail: leandromaciel@usp.br
#
#-----------------------------------------------------

# Carregar pacotes:

library(readxl)
library(forecast)
library(urca)

# Dados IPCA:

setwd("C:/Users/Leandro/Desktop/Disciplinas FEA-USP/EAD 830 - IA e ML/Aula 5")
DadosAula5 <- read_excel("DadosAula5.xls")

IPCA <- ts(DadosAula5$IPCA,start = c(2000,1),frequency = 12) # formato time series

plot(IPCA)
summary(IPCA)

# ----------------------------------------------------

# Simular processo raiz unit�ria:

raiz = 1
y = 3
for(i in 2:100){
  y[i] = raiz*y[i-1] + rnorm(1,0,1)
}
plot(y,type="l")

raiz = 0.3
y = 3
for(i in 2:100){
  y[i] = raiz*y[i-1] + rnorm(1,0,1)
}
plot(y,type="l")

# ----------------------------------------------------

# Teste ADF:

testeADF <- ur.df(y,type = "none",lags = 20,selectlags = "BIC")
summary(testeADF)

summary(ur.df(IPCA,type = "none",lags = 20,selectlags = "BIC"))
summary(ur.df(IPCA,type = "drift",lags = 20,selectlags = "BIC"))
summary(ur.df(IPCA,type = "trend",lags = 20,selectlags = "BIC"))

# S�rie � estacion�ria, caso fosse necess�rio diferencia��o:

y_dif <- diff(y)
plot(y_dif,type="l")
testeADF_dif <- ur.df(y_dif,type = "none",lags = 20,selectlags = "BIC")
summary(testeADF_dif)

# ----------------------------------------------------


# Dividir amostra em treino e teste:

n <- 24 # n�mero de meses para teste (previs�o)

IPCA_treino <- ts(IPCA[1:(length(IPCA)-n)],start = c(2000,1),frequency = 12)
IPCA_teste <- ts(IPCA[(length(IPCA)-n+1):length(IPCA)],start = c(2020,7),frequency = 12)

# Estimar modelo ARIMA:

modelo = Arima(IPCA_treino,order = c(2,0,2))
summary(modelo)  

# Ajuste:

plot(IPCA_treino)
lines(modelo$fitted,col="red")

# Res�duos:

hist(modelo$residuals,nclass = 15)
acf(modelo$residuals)
plot(modelo$residuals)

# Previs�o um passo � frente:

modelo2 = Arima(IPCA_teste,model = modelo)

# Ajuste:

accuracy(modelo2)

# Plots:

plot(IPCA_teste,type = "l")
lines(modelo2[["fitted"]],col="blue")

# Previs�o v�rios passos:

previsoes <- forecast(modelo,h = n)

# Ajuste:

accuracy(previsoes,IPCA_teste)

plot(previsoes)
lines(IPCA_teste,col="red")

# modelagem autom�tica auto.arima:

modeloAUTO <- auto.arima(IPCA_treino)

summary(modeloAUTO)  

# Ajuste:

plot(IPCA_treino)
lines(modeloAUTO$fitted,col="red")

modeloAUTO2 = Arima(IPCA_teste,model = modeloAUTO)

accuracy(modeloAUTO2)

plot(IPCA_teste,type = "l")
lines(modeloAUTO2[["fitted"]],col="blue")

# Previs�o v�rios passos:

previsoesAUTO = forecast(modeloAUTO,h=n)

plot(previsoesAUTO)
lines(IPCA_teste,col="red")

# -------------------------------------------------------------