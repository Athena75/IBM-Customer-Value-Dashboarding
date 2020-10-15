from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
import pandas as pd


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names].values

def make_full_pipeline(df):
    #df = pd.read_csv('../data/Customer-Value-Analysis.csv').set_index('Customer')
    X = df.drop(['Response'], axis=1)
    y = df.Response.apply(lambda X: 0 if X == 'No' else 1)

    cats = [var for var, var_type in X.dtypes.items() if var_type == 'object']
    nums = [var for var in X.columns if var not in cats]
    # Defining the steps in the categorical pipeline
    cat_pipeline = Pipeline([('cat_selector', FeatureSelector(cats)),
                             ('one_hot_encoder', OneHotEncoder(sparse=False))])

    # Defining the steps in the numerical pipeline
    num_pipeline = Pipeline([
        ('num_selector', FeatureSelector(nums)),
        ('std_scaler', StandardScaler()),
    ])

    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                                   ('cat_pipeline', cat_pipeline)]
                                 )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # fit and transform the custom transformer in train
    _ = full_pipeline.fit_transform(X_train)
    return full_pipeline