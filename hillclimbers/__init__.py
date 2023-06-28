import numpy as np
import pandas as pd
import plotly.express as px
from colorama import Fore, Back, Style


class partial:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        new_args = self.args + args
        new_kwargs = {**self.kwargs, **kwargs}
        return self.func(*new_args, **new_kwargs)

    
def climb_hill(
    train: pd.DataFrame = None, 
    oof_pred_df: pd.DataFrame = None, 
    test_pred_df: pd.DataFrame = None, 
    target: str = None, 
    objective: str = None, 
    eval_metric: partial = None,
    negative_weights: bool = False, 
    precision: float = 0.01, 
    plot_hill: bool = True, 
    plot_hist: bool = False,
    return_oof_preds: bool = False
) -> np.ndarray: # Returns test predictions (and oof predictions if return_oof_preds = True) resulting from hill climbing
    
    BOLD_TXT =   Style.BRIGHT
    YELLOW_TXT = BOLD_TXT + Fore.YELLOW
    GREEN_TXT =  BOLD_TXT + Fore.GREEN
    RED_TXT =    BOLD_TXT + Fore.RED
    BLUE_TXT =   BOLD_TXT + Fore.BLUE
    RESET_TXT =  Style.RESET_ALL
    
    STOP = False
    scores = {}
    i = 0
    
    hill_icon = f"{BLUE_TXT}   /\\  \n  /__\  hillclimbers{RESET_TXT}{BOLD_TXT} \n /    \\\n/______\\ \n{RESET_TXT}"
    print(hill_icon)

    oof_df = oof_pred_df
    test_preds = test_pred_df
        
    # Compute CV scores on the train data
    for col in oof_df.columns:
        scores[col] = eval_metric(train[target], oof_df[col])
        
    # Sort CV scores
    if objective == "minimize":
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=False)}
    elif objective == "maximize":
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    else:
        raise ValueError("Please provide valid hillclimbing objective (minimize or maximize)")
    
    print(f"{YELLOW_TXT}Models to be ensembled | ({len(scores)} total):{RESET_TXT} \n")
    max_model_len = max(len(model) for model in scores.keys())
    
    # Display models with their associated metric score
    for e, (model, score) in enumerate(scores.items()):
        model_padding = " " * (max_model_len - len(model))
        score_str = f"{score:.5f}".rjust(2)
        if e == 0:
            print(f"{GREEN_TXT}{model}:{model_padding} {score_str} (best solo model){RESET_TXT}")
        else:
            print(f"{BOLD_TXT}{model}:{model_padding} {score_str}{RESET_TXT}")
    print()

    oof_df = oof_df[list(scores.keys())]
    test_preds = test_preds[list(scores.keys())]
    current_best_ensemble = oof_df.iloc[:,0]
    current_best_test_preds = test_preds.iloc[:,0]
    MODELS = oof_df.iloc[:,1:].copy()
    history = [eval_metric(train[target], current_best_ensemble)]
    
    if precision > 0:
        if negative_weights:
            weight_range = np.arange(-0.5, 0.51, precision)
        else:
            weight_range = np.arange(precision, 0.51, precision)
    else:
        raise ValueError("precision must be a positive number")
        
    decimal_length = len(str(precision).split(".")[1])
    eval_metric_name = eval_metric.func.__name__
    
    print(f"{YELLOW_TXT}[Data preparation completed successfully] - [Initiate hill climbing]{RESET_TXT} \n")
    
    # Hill climbing
    while not STOP:
        
        i += 1
        potential_new_best_cv_score = eval_metric(train[target], current_best_ensemble)
        k_best, wgt_best = None, None
        
        for k in MODELS:
            for wgt in weight_range:
                potential_ensemble = (1 - wgt) * current_best_ensemble + wgt * MODELS[k]
                cv_score = eval_metric(train[target], potential_ensemble)
                
                if objective == "minimize":
                    if cv_score < potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt
                        
                elif objective == "maximize":
                    if cv_score > potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt

        if k_best is not None:
            current_best_ensemble = (1 - wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
            current_best_test_preds = (1 - wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
            MODELS.drop(k_best, axis = 1, inplace=True)
            
            if MODELS.shape[1] == 0:
                STOP = True
            
            if wgt_best > 0:
                print(f'{GREEN_TXT}Iteration: {i} | Model added: {k_best} | Best weight: {wgt_best:.{decimal_length}f} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}{RESET_TXT}')
            elif wgt_best < 0:
                print(f'{RED_TXT}Iteration: {i} | Model added: {k_best} | Best weight: {wgt_best:.{decimal_length}f} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}{RESET_TXT}')
            else:
                print(f'Iteration: {i} | Model added: {k_best} | Best weight: {wgt_best:.{decimal_length}f} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}')
                
            history.append(potential_new_best_cv_score)
            
        else:
            STOP = True
            
    if plot_hill:
        
        fig = px.line(x = np.arange(len(history)) + 1, y = history, color_discrete_sequence = ["#009933"], 
                      text = np.round(history, 5), labels = {"x": "Number of Models", "y": "CV"},
                      title = f"Cross Validation {eval_metric_name} vs. Number of Models with Hill Climbing",
                      template = "plotly_dark")
        
        # Used to avoid decimal places on the x axis (Number of models)
        x_tickvals = list(range(int((np.arange(len(history)) + 1).min()), int((np.arange(len(history)) + 1).max()) + 1))
        
        fig.update_traces(textposition = "top center", hovertemplate = "Number of Models: %{x}<br>CV: %{y}", textfont = {"size": 10},
                          marker = dict(size = 10, color = "#33cc33", line = dict(width = 1.5, color = "#FFFFFF")))
        
        fig.update_layout(autosize = False, width = 900, height = 500,
                          xaxis = {"tickmode": "array", "tickvals": x_tickvals},
                          xaxis_title = "Number of models", 
                          yaxis_title = f"{eval_metric_name}")      
        fig.show()
        
    if plot_hist:
        
        print()

        fig = px.histogram(x = current_best_test_preds, marginal = "box", color_discrete_sequence = ["#33cc33"],
                           title = f"Histogram of final test predicitons: {target}", template = "plotly_dark")
        
        fig.update_layout(autosize = False, width = 900, height = 500, xaxis_title = target) 
        fig.update_traces(hovertemplate = "test_pred: %{x}<br>count: %{y}")
        fig.show()
    
    if return_oof_preds: return current_best_test_preds.values, current_best_ensemble.values            
    else: return current_best_test_preds.values
