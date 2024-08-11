"""
One needs to install `LG_InfoTUtils` and its dependencies:

* install it from https://github.com/gerberl/LG_InfoTUtils. The following should
  be sufficient: `pip install git+https://github.com/gerberl/LG_MLUtils.git`.

* dependencies that would likely need to install in their local installation:
    - `disfunctools`: https://github.com/gerberl/disfunctools
    - `LG_MLUtils`: https://github.com/gerberl/LG_MLUtils

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample

from LG_InfoTUtils import (
    JS_div_ccX
)


def model_prediction_stability(model, X_train, X_test, cmp=JS_div_ccX, num_bootstraps=10):
    """
    In short, bootstrap-sample from X_train, retrain model m_i, and obtain
    predictions of m_i on X_test `num_boostraps` times; then, return a metric
    of divergence between the predicted value distributions.

    models: list

    returns what something like JS_div_ccX would:
    (mean_value, std_value, values, kdes)

    Example:
    --------
    >>> ames_full = pd.read_csv(
        'https://raw.githubusercontent.com/gerberl/6g7v0017-2122/main/datasets/ames/train.csv'
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
        
    >>> model_prediction_stability(rfr, X_train, X_test)


    CPU times: user 25 s, sys: 1.05 s, total: 26 s
    Wall time: 25.8 s
    (
        0.0020550776835121503,
        0.00137419705739459,
        array([0.00260392, 0.00159066, 0.00308427, 0.00132454, 0.00094505,
           0.00168814, 0.00135757, 0.00056038, 0.00534118]),
        [
            <scipy.stats._kde.gaussian_kde object at 0x159a68a90>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6b7d0>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6ad50>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6b290>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6a350>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6b710>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6aa10>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6add0>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6b090>,
            <scipy.stats._kde.gaussian_kde object at 0x159a6c2d0>
        ]
    )
    """

    predictions = np.empty((num_bootstraps, len(X_test))) 
    for i in range(num_bootstraps):
        X_train_resampled, y_train_resampled = resample(
            X_train, y_train
        )
        rfr.fit(X_train_resampled, y_train_resampled)
        predictions[i] = rfr.predict(X_test)

    # my metrics work in a column-wise fashion
    distributions = predictions.T

    # I am unpacking the values to remind myself of what the metric returns
    mean_value, std_value, values, kdes = JS_div_ccX(distributions)

    return (mean_value, std_value, values, kdes)


