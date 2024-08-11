"""
First go at this: 11-Aug-24. I am sure there is plenty of refactoring needed.

One needs to install `LG_InfoTUtils` and its dependencies:

* install it from https://github.com/gerberl/LG_InfoTUtils. The following should
  be sufficient: `pip install git+https://github.com/gerberl/LG_MLUtils.git`.

* dependencies that would likely need to install in their local installation:
    - `disfunctools`: https://github.com/gerberl/disfunctools
    - `LG_MLUtils`: https://github.com/gerberl/LG_MLUtils


Example:
--------
>>> ames_full = pd.read_csv(
    'https://raw.githubusercontent.com/gerberl/ML_Model_Stability/main/data/ames/train.csv'
)
>>> cols_of_interest = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'SalePrice'
]
>>> ames_sm = ames_full.loc[:, cols_of_interest]
>>> ames_sm['SalePrice'] = ames_sm['SalePrice']/1000

>>> X, y = ames_sm.drop(columns='SalePrice'), ames_sm['SalePrice']
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

>>> rfr = RandomForestRegressor(
    max_depth=8, min_samples_split=15, min_samples_leaf=8, n_estimators=400
)
>>> rfr.fit(X_train, y_train)
    
>>> mean_div, std_div, divs, kdes, Y_pred, _ = bootstrapped_model_prediction_stability(rfr, X_train, X_test)

>>> print(mean_div)
0.0015442675065352512

>>> print(values.shape)
(9,)

>>> print(values)
[0.0014414  0.00179891 0.00154782 0.00087064 0.00125398 0.00205539
 0.00141449 0.00164772 0.00186806]

>>> print(Y_pred.shape)
(292, 10)

# each column is a distribution for a model; rows are the X_test instances
>>> Y_pred[:5, :2]
array([[146.22845178, 136.21150308],
   [151.04972912, 150.299481  ],
   [171.90415304, 170.22567176],
   [227.57415478, 216.76904839],
   [190.91997288, 179.77918165]])

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.base import clone

from LG_InfoTUtils import (
    JS_div_ccX
)



def bootstrapped_model_prediction_stability(
        model, X_train, X_test, cmp=JS_div_ccX, num_bootstraps=10
    ):
    """
    In short, bootstrap-sample from X_train, retrain model m_i, and obtain
    predictions of m_i on X_test `num_boostraps` times; then, return a metric
    of divergence between the predicted value distributions.

    models: list

    returns what something like JS_div_ccX would (mean_value, std_value, values, kdes) as well as the model variants and their predicted values.

    Example:
    --------
    # Given existing RandomForestRegressor rfr and train and test data...

    >>> bootstrapped_model_prediction_stability(rfr, X_train, X_test)

    (
        0.0024152415009843805,
        0.0014502869385804222,
        array([0.00118089, 0.00119379, 0.00182453, 0.00237604, 0.00358754,
           0.00130437, 0.0009027 , 0.0042498 , 0.00511751]),
        [
            <scipy.stats._kde.gaussian_kde object at 0x17696d890>,
            <scipy.stats._kde.gaussian_kde object at 0x17696d550>,
            <scipy.stats._kde.gaussian_kde object at 0x17696d850>,
            <scipy.stats._kde.gaussian_kde object at 0x17696d9d0>,
            <scipy.stats._kde.gaussian_kde object at 0x17696df10>,
            <scipy.stats._kde.gaussian_kde object at 0x17696e050>,
            <scipy.stats._kde.gaussian_kde object at 0x17696e190>,
            <scipy.stats._kde.gaussian_kde object at 0x17696e350>,
            <scipy.stats._kde.gaussian_kde object at 0x17696e3d0>,
            <scipy.stats._kde.gaussian_kde object at 0x17696e590>
        ],
        array([[142.96614281, 132.51856678, 135.02081884, ..., 135.1775516 ,
            149.66795403, 139.22576473],
           [149.11558372, 151.81128986, 155.61121179, ..., 146.35321614,
            151.80135342, 147.82309865],
           [168.31450851, 171.95018105, 166.55004707, ..., 172.95269607,
            172.28959327, 168.71572942],
           ...,
           [168.13041755, 159.2104306 , 177.58982245, ..., 172.28052877,
            169.10345888, 167.26486345],
           [181.16399055, 186.33158772, 165.15364887, ..., 213.88432369,
            181.54954172, 200.83714306],
           [ 88.81910801,  85.86636109,  84.74612153, ...,  92.6606158 ,
             96.30901473,  92.37009859]]),
        [
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400),
            RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=15,
                          n_estimators=400)
        ]
    )
    """

    model_variants = bootstrap_models(model, X_train, num_bootstraps)
    Y_pred = get_Y_pred(model_variants, X_test)

    # I am unpacking the values to remind myself of what the metric returns
    mean_value, std_value, values, kdes = JS_div_ccX(Y_pred)

    # return distributions as well so that sampled/simulated values can be
    # looked at
    return (mean_value, std_value, values, kdes, Y_pred, model_variants)



def bootstrap_models(model, X_train, num_bootstraps=10):
    """
    """

    model_variants = []
    for i in range(num_bootstraps):
        model = clone(model)
        X_train_resampled, y_train_resampled = resample(
            X_train, y_train
        )
        model.fit(X_train_resampled, y_train_resampled)
        model_variants += [model]

    return model_variants



def get_Y_pred(models, X_test):
    """
    models: a list of model variants
    """

    # a list of distributions of predicted values, one for each model
    Y_pred = [ model.predict(X_test) for model in models ]
    # best to work with an array of predicted values; one column for each
    # distribution
    Y_pred = np.array(Y_pred).T

    return Y_pred



def plot_predicted_value_distributions(
        distributions, kdes, n_samples=1000, div=None, ax=None):
    """
    distributions are needed for the range of values
    kdes are needed for inferring densities at equally-spaced points

    Example:
    --------
    # take the above example for model_prediction_stability...
    ax = plot_predicted_value_distributions(distributions, kdes, div=mean_div)
    """

    min_x, max_x = np.min(distributions), np.max(distributions)
    # generate samples to have densities evaluated on
    x_plot = np.linspace(min_x, max_x, 1000)
    # each KDE is evaluated on the same set of x values
    ys_plot = [ kde(x_plot) for kde in kdes ]

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, sharey=True)

    for (i, y_plot) in enumerate(ys_plot):
        ax.plot(x_plot, y_plot, label=f"p{i+1}")
    ax.legend()

    if div:
        ax.text(
            0.01, 0.98, f'mean divergence: {div}',
            verticalalignment='top', horizontalalignment='left',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.5)
        )

    return ax