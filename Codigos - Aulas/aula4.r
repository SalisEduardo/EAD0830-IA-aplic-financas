# -------------------------------------------
# 
# RLM - CAPM e Modelo de 3 Fatores
# Prof. Leandro Maciel
# email: leandromaciel@usp.br
#
# -------------------------------------------

# Carregar os pacotes:

library(car)
library(readxl)

# Carregar dados:

Dados <- read_excel("Dados/DadosAula4.xlsx")


# Modelo CAPM Amazon:


capm = lm((R_AM-RF) ~ MktRF,data = Dados)
summary(capm)


# Modelo CAPM Amazon sem intercepto:

capm = lm((R_AM-RF) ~ 0 + MktRF,data = Dados)
summary(capm)

plot(capm$residuals)
mean(capm$residuals)

# Teste de heterocedasticidade de Breusch-Pagan:

ncvTest(capm)
# n�o rejeita H0 (homocedasticidade)



# Modelo 3F Amazon:


modeloff = lm((R_AM-RF) ~ MktRF + SMB + HML,data = Dados)
summary(modeloff)


# Modelo 3F Amazon sem intercepto:

modeloff = lm((R_AM-RF) ~ 0 + MktRF + SMB + HML,data = Dados)
summary(modeloff)

plot(modeloff$residuals)
mean(modeloff$residuals)

# Teste de heterocedasticidade de Breusch-Pagan:

ncvTest(modeloff)
# n�o rejeita H0 a 5% (homocedasticidade)

