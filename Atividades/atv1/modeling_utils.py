
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score, max_error, r2_score
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

from stationary_checks import *

def get_expanding_arima(train,test,print_flags=True):

    """
    train (pd.Series): train series indexed by date
    test (pd.Series): test series indexed by date
    """

    initial_data = train.copy() # n is the number of steps you want to forecast
    expanding_arima_predictions = []

    models_params = []

    for i in range(len(test)):

        
        dt_value = test.reset_index().iloc[i,0]
        series_value = test.reset_index().iloc[i,1]

        

        model = pm.auto_arima(y=initial_data,
                            start_p=1,
                            start_q=1,
                            test='adf',  
                            max_p=10, 
                            max_d=9, 
                            max_q=10,
                            error_action='ignore',
                            information_criterion = 'aic', # criterio para selecionar melhor modelo
                            suppress_warnings=True,
                            maxiter=100, 
                            stepwise=True
                            )
        
        models_params.append(model.order)
        
        model_fitted = ARIMA(initial_data,order=model.order).fit()

        single_forecast = model_fitted.forecast(steps=1)

        expanding_arima_predictions.append({"Date":dt_value,
                                            "Expanding_ARIMA_pred":single_forecast[0] })
        initial_data[dt_value] = series_value

        if print_flags:
            print(dt_value, "forecasted !!!!")
            print("Prediction: ", single_forecast[0] )
            print("-"*15)


    expanding_arima_predictions = pd.DataFrame(expanding_arima_predictions)

    results = {"parameters": models_params,
               "expand_forecasts":expanding_arima_predictions}
    
    return results





def checks_stats(series,**kwargs):
    print("Unit root test: ","\n","\n",check_unit_root(series,  confidence=0.05),"\n","\n","-"*50)
    print("Trend Test:","\n","\n",check_trend(series,confidence=0.05),"\n","\n","-"*50)
    print("Seasonality Test: ","\n","\n",check_seasonality(series, confidence=0.05, max_lag=720,seasonal_period=None) ,"\n","\n","-"*50) # two years of lags 
    print("Heteroscedastisticity: ","\n","\n",check_heteroscedastisticity(series,confidence=0.05),"\n","\n","-"*50)


def plot_acf_pcf(y,fig_title='ACF and PACF Plots'):
    with plt.style.context("seaborn-deep"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # ACF plot
        sm.graphics.tsa.plot_acf(y, lags=20, ax=ax[0])
        ax[0].set_xlabel('lags')
        ax[0].set_ylabel('corr')
        ax[0].set_title('ACF Plot')

        # PACF plot
        sm.graphics.tsa.plot_pacf(y, lags=20, ax=ax[1])
        ax[1].set_xlabel('lags')
        ax[1].set_ylabel('corr')
        ax[1].set_title('PACF Plot')

        fig.suptitle(fig_title, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the figure title
        plt.show()
                

def train_test_split(df, split=0.8,date_column ="date",by_date=False,print_ranges=True):
    """
    Performs train-test split on the input df

    Args:
        df (pd.DataFrame): The input data
        split (float): The fraction of the data to use for training

    Returns:
        tuple: A tuple containing the training and testing data
    """

    df = df.sort_values(date_column)
    df[date_column] = pd.to_datetime(df[date_column])

    if by_date:
        train_df = df[df[date_column] < split]
        test_df = df[df[date_column] > split]
    else:
        split_index = int(len(df) * split)
        train_df = df[:split_index]
        test_df = df[split_index:]


    train_df = train_df.set_index(date_column)
    test_df = test_df.set_index(date_column)


    if print_ranges:
        print(f"Train : start:{train_df.index.min()} ---- end:{train_df.index.max()}")
        print(f"Test : start:{test_df.index.min()} ---- end:{test_df.index.max()}")

    return train_df, test_df


def get_kpis_summary(y_true, y_pred):

    """
    Get a table summarizing the results of a model

    Args:
        y_true (pd.Series , list , np.array): test values
        y_pred (pd.Series , list , np.array): values predicted by some model

    Returns:
        np.array: An array containing the predicted values for the next 30 time steps
    """


    mae = mean_absolute_error(y_true, y_pred)

    # calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # calculate the Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # calculate the Mean Absolute Scaled Error (MASE)
    mase = mae / np.mean(np.abs(np.diff(y_true)))

    # calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # calculate the Overall Percentage Error (OPE)
    ope = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))

    # calculate the Explained Variance Score
    evs = explained_variance_score(y_true, y_pred)

    # calculate the Maximum Error
    max_err = max_error(y_true, y_pred)

    # calculate the R-squared (Coefficient of determination)
    r2 = r2_score(y_true, y_pred)


    kpis = pd.DataFrame([
        {"KPI":"MSE","Result":mse},
        {"KPI":"RMSE","Result":rmse},
        {"KPI":"MAE","Result":mae},       
        {"KPI":"MASE","Result":mase},
        {"KPI":"OPE","Result":ope},
        {"KPI":"MAPE","Result":mape},
        {"KPI":"EVS","Result":evs},
        {"KPI":"Max_err","Result":max_err},
        {"KPI":"Rsquared","Result":r2}
        
        ])

    return kpis


def check_arima_residuals(arima_fitted,aucorr_lags=20,plot_hist=True,plot_qq=True):
    
    residuals = arima_fitted.resid
    mu_residuals = residuals.mean()
    lb_test = acorr_ljungbox(residuals,lags=[aucorr_lags]) #no autocorr up to 10 levels
    lb_test_pvalues = lb_test["lb_pvalue"].values[0]
    has_resid_autcorr = lb_test_pvalues < 0.05

    stat_shapiro, shapiro_pvalue = shapiro(residuals)

    is_norm_resid = shapiro_pvalue >  0.05

    results = {"residuals": residuals,
                "lb_test" : lb_test,
                "lb_test_pvalues" : lb_test_pvalues,
                "has_resid_autcorr" : has_resid_autcorr,
                "stat_shapiro":stat_shapiro, 
                "shapiro_pvalue": shapiro_pvalue, 
                "is_norm_resid" : is_norm_resid}
    
    if plot_hist:
        plt.hist(residuals, bins=50)
        plt.title('Histogram of Residuals')
        plt.show()

    if plot_qq:
        

        sm.qqplot(residuals, line='s')
        plt.title('Q-Q Plot of Residuals')
        plt.show()

    return results