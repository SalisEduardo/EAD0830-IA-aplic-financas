library(wooldridge)
library(tidyverse)
library(tidyr)
library(dplyr)
library(glmnet)
library(readxl)
library(xts)
library(zoo)
library(PerformanceAnalytics)

setwd("~/Documents/Faculdade/EAD0830 - IA e ML Aplicados a Financas/Pratica/Fatores")


# Brazilian Factors 

factor_tables <- list.files("Dados",pattern = '_Factor')
factor_tables <- append(factor_tables,"Risk_Free.xls")

factor_tables <-paste("Dados/",factor_tables,sep='')

format_factor_tables <- function(path){
  df <- read_xls(path) |>
    dplyr::mutate(date = as.Date(paste(year,month,day,sep='-'))) |>
    dplyr::select(-c(year,month,day)) |>
    dplyr::select(date,everything())
  return(df)
}


tabsfactors_list <- lapply(factor_tables, format_factor_tables)

nefin_factors <-  tabsfactors_list |> 
  purrr::reduce(left_join,by='date')


# Indie Master FIA

indie <- read.csv("Indie.csv") |> 
  dplyr::mutate(date= as.Date(date),Indie= as.numeric(Indie)) |> 
  dplyr::arrange(date) |>
  as_tibble()


indie_returns <- xts(indie[-1],order.by = indie$date) |>
  CalculateReturns() |> 
  fortify.zoo() |> 
  as.tibble() |>
  rename("date" = "Index")


indie_factors <- indie_returns |> 
  dplyr::left_join(nefin_factors,by='date') |>
  tidyr::drop_na() 

factor_model <- lm((Indie-Risk_free) ~ 0+ Rm_minus_Rf  + SMB + HML + WML+ IML,
                   data = indie_factors)


summary(factor_model)





