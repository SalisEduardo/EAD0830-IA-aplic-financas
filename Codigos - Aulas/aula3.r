
# -------------------------------------------
# 
# RLM
# Prof. Leandro Maciel
# email: leandromaciel@usp.br
#
# -------------------------------------------

# Carregar os pacotes:

library(wooldridge)
library(car)

# Dados

dados <- beauty

# Estimar o modelo:

modelo <- lm(lwage ~ educ + exper + belavg + abvavg + female, data = beauty)

# Exibir principais resultados:

summary(modelo)

# Correlacoes:

cor(beauty$lwage,beauty$exper)
cor(beauty$lwage,beauty$educ)

# Plot residuos:

plot(modelo$residuals)
mean(modelo$residuals)


# Teste de heterocedasticidade de Breusch-Pagan:

ncvTest(modelo)
# Nao rejeita H0 (homocedasticidade)

hccm(modelo)

Sign in to access your highlights
Login / Signup
weava logo
Drop here!