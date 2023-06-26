import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score ,confusion_matrix ,precision_score, recall_score, f1_score, classification_report ,roc_curve, roc_auc_score ,roc_curve, auc, ConfusionMatrixDisplay , RocCurveDisplay
from matplotlib import pyplot as plt

def get_metrics(model,y_true,X_true,probs=None,predictions=None):
    if predictions is None:
        y_pred =  model.predict(X_true)
    else:
        y_pred = predictions

    if probs is None:
        y_pred_proba =  model.predict_proba(X_true)
    else:
        y_pred_proba = probs

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:,-1] )
    auc = roc_auc_score(y_true, y_pred_proba[:,-1] )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    metrics = {
        'Accuracy' : accuracy,
        'Precision' : precision,
        'Recall' : recall,
        'F1-Score' : f1,
        'False Positive Ratio': fpr,
        'True Positive Ratio':tpr,
        'Thresholds':thresholds,
        'Area Under the Curve': auc,
        "Gini": round(2*auc-1,2),
        'Confussion Matrix' : cm,
        'Classification Report' : report

    }

    return metrics



def get_metrics_table(model,y_true,X_true,model_name,probs=None,predictions=None):
    if predictions is None:
        y_pred =  model.predict(X_true)
    else:
        y_pred = predictions

    if probs is None:
        y_pred_proba =  model.predict_proba(X_true)
    else:
        y_pred_proba = probs

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:,-1] )
    auc = roc_auc_score(y_true, y_pred_proba[:,-1] )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    metrics = {
        "Model":model_name,
        'Accuracy' : accuracy,
        'Precision' : precision,
        'Recall' : recall,
        'F1-Score' : f1,
        'Area Under the Curve': auc,
        "Gini": round(2*auc-1,2)

    }

    df = pd.DataFrame([metrics])

    return df






def  display_metrics(train_metrics_report,test_metrics_report,not_show=['Confussion Matrix','Classification Report','False Positive Ratio','True Positive Ratio','Thresholds']):
    for k in train_metrics_report.keys():
        if k not in not_show:
            print(k, " - Train : ", round(train_metrics_report[k],4))
            print(k, " - Test : ", round(test_metrics_report[k],4))
            print("-"*100)

def plot_classification_metrics(model,y_true,X_true):
    y_pred =  model.predict(X_true)
    fig, (ax1, ax2)= plt.subplots(1,2, figsize=(12, 6))
    RocCurveDisplay.from_estimator(model, X_true, y_true).plot(ax=ax2)

    ax2.plot([0,1],[0,1],'k--',label='Benchmark')
    ax2.set_title('ROC Curve Prediction')
    ax2.legend()
    
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    ConfusionMatrixDisplay(confusion_matrix = conf_matrix).plot(ax=ax1)

    plt.close()

