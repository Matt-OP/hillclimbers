# hillclimbers

![hillclimber](https://github.com/Matt-OP/hillclimbers/assets/105291392/3d2ef229-a3c2-44d2-90a8-4d83e9ab43d4)

A python module that uses hill climbing to iteratively blend machine learning model predictions. 

Hill climbing attempts to maximize (or minimize) a target function $f(x)$. At each iteration, hill climbing will adjust a single element in $x$ and determine whether the change improves the value of $f(x)$. With hill climbing, any change that improves $f(x)$ is accepted, and the process continues until no change can be found to improve the value of $f(x)$. Then $x$ is said to be "locally optimal". Hill climbing will not necessarily find the global maximum/minimum, but may instead converge on a local maximum/minimum.

The models with the best cross validation scores are not always chosen first. Instead hill climbing chooses diverse models.

"I love hill climbing because it can take lots of models and picks the best small subset of models. (i.e. its like Lasso regression) And it computes ensemble model weights." - [Chris Deotte](https://www.kaggle.com/cdeotte) (Kaggle legend)

hillclimbers was used to achieve 4th place in Kaggle's Season 3, Episode 14 Playground Series competition - Prediction of Wild Blueberry Yield. I created a [discussion post](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/410639) that provides a detailed explanation of the solution and also shared the [notebook](https://www.kaggle.com/code/mattop/4th-place-hillclimbing-with-5-models-ps-s3-e14) that was used as the final submission which showcases hillclimbers.

```python
!pip install hillclimbers

from hillclimbers import climb_hill, partial
```
```python
def climb_hill(
    train=None, 
    oof_pred_df=None, 
    test_pred_df=None, 
    target=None, 
    objective=None, 
    eval_metric=None,
    negative_weights=False, 
    precision=0.01, 
    plot_hill=True, 
    plot_hist=False
    return_oof_preds=False
) -> np.ndarray: # Returns test predictions resulting from hill climbing
```

### Parameters:

- `train`: pd.DataFrame | Training dataset that was used to train the models to be blended using hillclimbing. Used to compute the cross validation scores of the Out-Of-Fold (OOF) predictions in `oof_pred_df`.

- `oof_pred_df`: pd.DataFrame | A single DataFrame of Out-Of-Fold (OOF) predictions to be blended using hillclimbing with n columns for n models. This can be achieved easily using `pd.concat([oof_pred_df1, oof_pred_df2, ...], axis=1)` or by using a DataFrame constructor. Each column name should be the name of the model, for example: `XGBRegressor` or simply `XGB`. 

- `test_pred_df`: pd.DataFrame | A single DataFrame of test set predictions to be blended using hillclimbing with n columns for n models. This can be achieved easily using `pd.concat([test_pred_df1, test_pred_df2, ...], axis=1)` or by using a DataFrame constructor. Each column name should be the name of the model, for example: `XGBRegressor` or simply `XGB`. NOTE: If you are using hillclimbers for a Kaggle competiton, you do not need to remove the `id` column(s) in `test_pred_df`.

- `target`: string | Represents the name of the target column in the train set you are prediciting. 

- `objective`: string, {"minimize", "maximize"} | Specifies the goal of the evaluation metric. For example the `objective` should be set to `minimize` if using an evaluation metric such as RMSE (Root Mean Squared Error) and set to `maximize` if using an evaluation metric such as ROC AUC (Receiver Operating Characteristic, Area Under The Curve). 

- `eval_metric`: partial | Evaluation metric to optimize. Since evaluation metrics are functions themselves, you must pass it as partial function. This can be achieved using hillclimbers' built in `partial` or using the `from functools import partial` implementation. The following is an example of the correct format: `partial(mean_squared_log_error, squared=True, ...)`

- `negative_weights`: bool | Turn `negative_weights` on or off. Using negative weights will increase computation time because of the increased size of the weight array that will be iterated through. "If the train data is small and we are afraid of overfitting, then restricting to only positive weights is safer and more likely to generalize." - Chris Deoette

- `precision`: float | Specifies the step to be taken in the array of weights. The default value of `0.01` should be sufficient in most cases and the recommended value falls between `0.01` - `0.001` (values lower than `0.001` will result in an extremely long computation process)

- `plot_hill`: bool | Plots a lineplot of the cross validation scores as each model is iteratively added to the ensemble.

- `plot_hist`: bool | Plots a histogram of the test predictions resulting from hill climbing.

- `return_oof_preds`: bool | Returns the ensemble of Out-Of-Fold (OOF) predictions which is computed using the weights optimized for the evaluation metric. Please note that activating `return_oof_preds` will change the number of arrays retrieved from 1 to 2, with the test predictions being the first array and the OOF predictions being the second array.
