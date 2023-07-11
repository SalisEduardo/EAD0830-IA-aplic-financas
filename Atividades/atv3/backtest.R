library(PerformanceAnalytics)
library(PortfolioAnalytics)
library(ROI.plugin.quadprog)
library(ROI.plugin.glpk)
library(doParallel)
library(tidyverse)
library(quantmod)
library(parallel)
library(plot3D)
library(readxl)
library(dplyr)
library(GA)


# Composicao do IBOV no dia 2022-12-29 (investidor monatndo a carteira em janeiro de 2023)
core_ibov_tickers <- readxl::read_excel('COMDINHEIRO_IBOV29122023.xlsx') |> 
  dplyr::top_n(15,Peso) |>
  dplyr::mutate(Ticker = paste(Ticker,'.SA',sep='')) |>
  dplyr::pull(Ticker)

charlesRiver <-readxl::read_excel("COMDINHEIRO_carteira_charlesRiverJan2023.xlsx")
colnames(charlesRiver) <- c('ativo','desc','compras','vendas','final','peso')
view(charlesRiver)

not_stocks <- c("Investimento no Exterior",'CONTA CORRENTE NO EXTERIOR','VALORES A PAGAR','VALORES A RECEBER','NTNB_10012018_15082028')

stocks_CR <-  charlesRiver |>
  dplyr::filter(ativo %in% not_stocks == FALSE) |>
  dplyr::mutate(ativo = ifelse(desc =='Investimento no Exterior',ativo,paste(ativo,".SA",sep='') ) ) |>
  dplyr::group_by(ativo) |>
  dplyr::summarise(peso=sum(peso))|>
  dplyr::top_n(15,peso) |>
  dplyr::pull(ativo)

selected_tickers<- stocks_CR

getSymbols(selected_tickers, src = "yahoo", from ='2021-01-01', to='2023-05-31' ) # not run more than one time

start_date <- as.Date('2021-01-01')
end_date <- as.Date('2023-05-31')

getAdjustedClose <- function(symbol) {
  getSymbols(symbol, from = start_date, to = end_date)
  
  adjusted_close <- Ad(get(symbol))
  
  return(adjusted_close)
}

# Retrieve Adjusted Close prices for each ticker
adjusted_close_list <- lapply(selected_tickers, getAdjustedClose)



stocks_prices <- do.call(cbind, adjusted_close_list)

stocks_prices_df <- stocks_prices |> as.data.frame()


stocks_prices_df$date <- rownames(stocks_prices_df)

stocks_prices_df <-cbind(stocks_prices_df['date'],stocks_prices_df[colnames(stocks_prices)])

stocks_prices_df |> write.csv('stock_prices.csv')

colnames(stocks_prices) <- gsub("\\.SA.Adjusted$", "", colnames(stocks_prices))
colnames(stocks_prices) <- gsub("\\.Adjusted$", "", colnames(stocks_prices))



stocks_daily_returns <- stocks_prices |> 
  PerformanceAnalytics::Return.calculate(method = 'log') |> 
  na.omit()


chart.CumReturns(stocks_daily_returns$NEXA)

dol_NEXA <- stocks_daily_returns$NEXA

# Ajusting NEXA returns to BRL - no working

currency_pair <- "USDBRL=X"
getSymbols(currency_pair, src = "yahoo", from = start_date, to = end_date)
exchange_rates <- Cl(get(currency_pair))
exchange_rate_returns <- dailyReturn(exchange_rates, type = "log")
exchange_rate_returns <- exchange_rate_returns[-1]

brl_nexa <- stocks_daily_returns$NEXA * exchange_rate_returns

chart.CumReturns(stocks_daily_returns$NEXA * exchange_rate_returns)

stocks_daily_returns$NEXA <- dol_NEXA
 
stocks_daily_returns$NEXA <- NULL

training_portfolio <- stocks_daily_returns['2021/2022']
validation_portfolio <- stocks_daily_returns['2023']


# Gradient Optimization

## Invest all capital, Shot allowed , minimize standard deviation - Global Minimum Variation Portfolio
gmvp_specs <-portfolio.spec(colnames(stocks_daily_returns)) |>
  #add.constraint(type = "long_only") |> 
  add.constraint(type = "full_investment") |> 
  add.objective(type = "risk", name = "StdDev",risk_aversion=9999)


gmvp_ROI <- optimize.portfolio(R = training_portfolio, 
                               portfolio = gmvp_specs, 
                               optimize_method = "ROI",trace = TRUE)

weights_gmv_roi <- extractWeights(gmvp_ROI)

out_sample_gmv_ROI <- Return.portfolio(R = validation_portfolio,
                                       weights = weights_gmv_roi)

charts.PerformanceSummary(out_sample_gmv_ROI,main = "Global Minimun Variation with ROI")


daily_cdi <- (1 + 0.1365)^(1/252) - 1

kpis_gmv_ROI <- table.AnnualizedReturns(out_sample_gmv_ROI,Rf=daily_cdi)



# Genetic Algorithms 

# Constrains
constraint = function(weights) {
  boundary_constr = (sum(weights)-1)**2   # "sum x = 1" constraint
  return (boundary_constr)
}

# Define the objective function
portfolioObjective <- function(weights) {
  portfolioReturn <- sum(weights * colMeans(training_portfolio))
  portfolioStdDev <- sqrt(t(weights) %*% cov(training_portfolio) %*% weights)
  return(-portfolioStdDev - 100 * constraint(weights))  # Minimize standard deviation, so negative sign
}



# Set up the GA optimization

gaResult <- ga(type = "real-valued",
               fitness = portfolioObjective, #objective
               lower = rep(-1, ncol(training_portfolio)),  # Lower bounds for weights (allow negative)
               upper = rep(1, ncol(training_portfolio)),  # Upper bounds for weights,
               popSize = 100,  # Population size
               maxiter = 5000,  # Maximum number of iterations
               pmutation = 0.1, # default mutation rate
               elitism = max(1, round(50*0.05)), # default - 5% popSize
               run = 100,  # Number of runs
               parallel = TRUE, # parallel processing
               monitor = TRUE, # print the evolutions
               seed = 42)  # Set a seed for reproducibility

# Apresentar resultados principais:

summary(gaResult)
weights_gmv_GA <- as.vector(summary(gaResult)$solution)


names(weights_gmv_GA) <- colnames(stocks_daily_returns)

out_sample_gmv_GA <- Return.portfolio(R = validation_portfolio,
                                       weights = weights_gmv_GA)

charts.PerformanceSummary(out_sample_gmv_GA,main = "Global Minimun Variation with ROI")


kpis_gmv_GA <- table.AnnualizedReturns(out_sample_gmv_GA,Rf=daily_cdi)



# Comparison

kpis_comparison <- cbind(kpis_gmv_GA,kpis_gmv_ROI)
colnames(kpis_comparison) <- c("GA","ROI_Gradient")




comparison_daily_returns <- cbind(out_sample_gmv_GA,out_sample_gmv_ROI)
colnames(comparison_daily_returns) <- c("GA","ROI_Gradient")




PerformanceAnalytics::charts.PerformanceSummary(comparison_daily_returns, colorset=rich6equal,
                                                lwd=2, cex.legend = 1.0, event.labels = TRUE, main = "")



