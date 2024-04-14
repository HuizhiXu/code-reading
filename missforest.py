from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# X = pd.DataFrame(
#     {
#         "A": [np.nan, np.nan, 3],
#         "B": [1, 2, 3],
#         "C": [4, 5, 6],
#         "D": [7, 8, 9],
#         "E": [10, np.nan, 3],
#         "F": [1, 2, 34],
#         "G": ["male", "male", "male"],
#         "H": ["large", np.nan, "small"],
#         "I": ["cat", "dog", "dog"],
#     }
# )





def test_missing_values(X):
    """
    checks if there is any column with missing values in all rows by returning a boolean value.


    """
    print("Testing missing values...")
    if np.any(X.isnull().sum(axis=0) >= (X.shape[0])):
        raise ValueError("One or more columns have all rows missing.")


def _check_if_all_single_type(X):
    """
    check 同一个feature里面的数据类型是否一致

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.
    """

    vectorized_type = np.vectorize(type)
    for c in X.columns:
        feature_no_na = X[c].dropna()
        all_type = vectorized_type(feature_no_na)
        all_unique_type = pd.unique(all_type)
        n_type = len(all_unique_type)
        if n_type > 1:
            print(f"Feature {c} has more than one " f"datatype.")


def _get_obs_row(X):
    """
    Class method '_get_obs_row' gather the rows of any rows that do not
    have any missing values.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    Return
    ------
    None
    """
    not_empty_mask = X.notna().all(axis=1)
    # 使用布尔值的DataFrame过滤原始DataFrame，只保留非空行
    _obs_row = X[not_empty_mask]

    return _obs_row


def _get_missing_rows(X):
    """
    Class method '_get_missing_rows' gather the index of any rows that has
    missing values.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    Return
    ------
    miss_row : dict
    Dictionary that contains features which has missing values as keys, and
    their corresponding indexes as values.
    """

    mask = X.isna().any(axis=1)  # 标记包含空值的行
    _miss_row = X[mask]  # 选择标记为True的行

    return _miss_row


def _get_missing_cols(X):
    """
    Class method '_get_missing_cols' gather the columns of any rows that
    has missing values.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    Return
    ------
    None
    """
    _miss_row = {}
    _missing_cols = None

    is_missing = X.isnull().sum(axis=0).sort_values() > 0
    _missing_cols = X.columns[is_missing]

    return _missing_cols


def _get_map_and_rev_map(X, categorical):
    """
    Class method '_get_map_and_rev_map' gets the encodings and the reverse
    encodings of categorical variables.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    categorical : list
    All categorical features of X.

    Return
    ------
    None
    """
    _mappings = {}
    _rev_mappings = {}
    for c in X.columns:
        if c in categorical:
            unique = X[c].dropna().unique()
            n_unique = range(X[c].dropna().nunique())

            _mappings[c] = dict(zip(unique, n_unique))
            _rev_mappings[c] = dict(zip(n_unique, unique))

    return _mappings, _rev_mappings


def _get_initials(X, categorical, initial_guess="mean"):
    """
    Class method '_initial_imputation' calculates and stores the initial
    imputation values of each features in X.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    categorical : list
    All categorical features of X.

    Return
    ------
    None
    """
    _initials = {}
    intersection = set(categorical).intersection(set(X.columns))
    if not intersection == set(categorical):
        raise ValueError(
            "Not all features in argument 'categorical' " "existed in 'X' columns."
        )

    for c in X.columns:
        if c in categorical:
            _initials[c] = X[c].mode().values[0]
        else:
            if initial_guess == "mean":
                _initials[c] = X[c].mean()
            elif initial_guess == "median":
                _initials[c] = X[c].median()
            else:
                raise ValueError(
                    "Argument 'initial_guess' only accepts " "'mean' or 'median'."
                )

    return _initials

def _initial_imputation(X, categorical):
    """Class method '_initial_imputation' imputes the values of features
    using the mean or median if they are numerical variables, else, imputes
    with mode.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    Return
    ------
    X : pd.DataFrame of shape (n_samples, n_features)
    Imputed Dataset (features only).
    """
    _initials = _get_initials(X, categorical)
    for c in X.columns:
        X[c].fillna(_initials[c], inplace=True)

    return X



def fit(X, categorical=None):
    X = X.copy()

    # make sure 'X' is either pandas dataframe, numpy array or list of
    # lists.
    if (
        not isinstance(X, pd.DataFrame)
        and not isinstance(X, np.ndarray)
        and not (isinstance(X, list) and all(isinstance(i, list) for i in X))
    ):
        raise ValueError(
            "Argument 'X' can only be pandas dataframe, numpy" " array or list of list."
        )

    # if 'X' is a list of list, convert 'X' into a pandas dataframe.
    if isinstance(X, np.ndarray) or (
        isinstance(X, list) and all(isinstance(i, list) for i in X)
    ):
        X = pd.DataFrame(X)

    # make sure 'categorical' is a list of str.
    if (
        categorical is not None
        and not isinstance(categorical, list)
        and not all(isinstance(elem, str) for elem in categorical)
    ):
        raise ValueError(
            "Argument 'categorical' can only be list of " "str or NoneType."
        )

    # make sure 'categorical' has at least one variable in it.
    if categorical is not None and len(categorical) < 1:
        raise ValueError(f"Argument 'categorical' has a len of " f"{len(categorical)}.")

    # Check for +/- inf
    if categorical is not None and np.any(np.isinf(X.drop(categorical, axis=1))):
        raise ValueError("+/- inf values are not supported.")

    # make sure there is no column with all missing values.
    if np.any(X.isnull().sum() == len(X)):
        raise ValueError("One or more columns have all rows missing.")

    # initials = {}
    # _miss_row = {}
    # _missing_cols = None
    # self._obs_row = None
    # self._mappings = {}
    # self._rev_mappings = {}

    if categorical is None:
        categorical = []
    categorical = categorical
    numerical = [c for c in X.columns if c not in categorical]
    print(categorical, numerical)

    _check_if_all_single_type(X)
    _miss_row = _get_missing_rows(X)
    _missing_cols = _get_missing_cols(X)
    _obs_row = _get_obs_row(X)
    _mappings, _rev_mappings = _get_map_and_rev_map(X, categorical)
    _initials = _get_initials(X, categorical)
    _initial_imputation(X, categorical)

    X_imp = _label_encoding(X, _mappings)   
    print(_miss_row)
    
    all_gamma_cat = []
    all_gamma_num = []
    n_iter = 0

    rf_regressor = RandomForestRegressor(
                )
    
    rf_classifier = RandomForestClassifier(
               )
    
    
    for c in _missing_cols:
        if c in _mappings:
            estimator = deepcopy(rf_classifier)
        else:
            estimator = deepcopy(rf_regressor)

        # Fit estimator with imputed X
        # X_obs = X_imp.drop(c, axis=1).loc[_obs_row]
        # y_obs = X_imp[c].loc[_obs_row]
        X_obs = X_imp.drop(c, axis=1)
        y_obs = X_imp[c]

        estimator.fit(X_obs, y_obs)
        print(X_obs)
            # estimator.fit(X_obs, y_obs)

        # Predict the missing column with the trained estimator
        miss_index = _miss_row[c]
        X_missing = X_imp.loc[miss_index]
        X_missing = X_missing.drop(c, axis=1)
        y_pred = estimator.predict(X_missing)
        y_pred = pd.Series(y_pred)
        y_pred.index = _miss_row[c]

    #         # Update imputed matrix
    #         X_imp.loc[miss_index, c] = y_pred

    #         self._all_X_imp_cat.append(X_imp[self.categorical])
    #         self._all_X_imp_num.append(X_imp[self.numerical])

    #     if len(self.categorical) > 0 and len(self._all_X_imp_cat) >= 2:
    #         X_imp_cat = self._all_X_imp_cat[-1]
    #         X_imp_cat_prev = self._all_X_imp_cat[-2]
    #         gamma_cat = (
    #                 (X_imp_cat != X_imp_cat_prev).sum().sum() /
    #                 len(self.categorical)
    #         )
    #         all_gamma_cat.append(gamma_cat)

    #     if len(self.numerical) > 0 and len(self._all_X_imp_num) >= 2:
    #         X_imp_num = self._all_X_imp_num[-1]
    #         X_imp_num_prev = self._all_X_imp_num[-2]
    #         gamma_num = (
    #                 np.sum(np.sum((X_imp_num - X_imp_num_prev) ** 2)) /
    #                 np.sum(np.sum(X_imp_num ** 2))
    #         )
    #         all_gamma_num.append(gamma_num)

    #     n_iter += 1
    #     if n_iter > self.max_iter:
    #         break

    #     if (
    #             n_iter >= 2 and
    #             len(self.categorical) > 0 and
    #             all_gamma_cat[-1] > all_gamma_cat[-2]
    #     ):
    #         break

    #     if (
    #             n_iter >= 2 and
    #             len(self.numerical) > 0 and
    #             all_gamma_num[-1] > all_gamma_num[-2]
    #     ):
    #         break

    # # mapping the encoded values back to its categories.
    # X = self._rev_label_encoding(X_imp, self._rev_mappings)

    # return X






def _label_encoding(X, mappings):
    """
    Class method '_label_encoding' performs label encoding on given
    features and the input mappings.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    mappings : dict
    Dictionary that contains the categorical variables as keys and their
    corresponding encodings as values.

    Return
    ------
    X : X : pd.DataFrame of shape (n_samples, n_features)
    Label-encoded dataset (features only).
    """

    for c in mappings:
        X[c] = X[c].map(mappings[c]).astype(int)

    return X



def _integer_encoding(X,cat_vars=None):

    from sklearn.preprocessing import LabelEncoder
    key_label_mapping = {}
    categorical = [X.columns[index] for index in cat_vars]


    for c in categorical:
        le = LabelEncoder()
        mask = X.loc[:,c].isna()
        X.loc[:,c] = le.fit_transform(X.loc[:,c])
        X.loc[:,c] = X.loc[:,c].mask(mask,np.nan)

                # Get the mapping from encoded labels to original labels
        label_mapping = dict(zip(range(len(le.classes_)), le.classes_))

        key_label_mapping[c] = label_mapping
        # Print the label mapping dictionary
        print(label_mapping)

    return X,key_label_mapping


    """
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
    """
 





    # >>> le.transform(["tokyo", "tokyo", "paris"])
    # array([2, 2, 1]...)
    # >>> list(le.inverse_transform([2, 2, 1]))
    # ['tokyo', 'tokyo', 'paris']



if __name__ == "__main__":
#     X = pd.DataFrame(
#     {
#         "A": [np.nan, np.nan, 3, np.nan, np.nan, 3, np.nan, np.nan, 3, np.nan, np.nan, 3, np.nan, np.nan, 3, np.nan, np.nan, 3, np.nan, np.nan, 3],
#         "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
#         "C": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
#         "D": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#         "E": [10, np.nan, 3, 10, np.nan, 3, 10, np.nan, 3, 10, np.nan, 3, 10, np.nan, 3, 10, np.nan, 3, 10, np.nan, 3],
#         "F": [1, 2, 34, 1, 2, 34, 1, 2, 34, 1, 2, 34, 1, 2, 34, 1, 2, 34, 1, 2, 34],
#         "G": ["male", "male", "male", np.nan, "female", "female", "male", "male", "male", "female", "female", "female", "male", "male", "male", "female", "female", "female", "male", "male", "male"],
#         "H": ["large", np.nan, "small", "large", np.nan, "small", "large", np.nan, "small", "large", np.nan, "small", "large", np.nan, "small", "large", np.nan, "small", "large", np.nan, "small"],
#         "I": ["cat", "dog", "dog", "cat", "dog", "dog", "cat", "dog", "dog", "cat", "dog", "dog", "cat", "dog", "dog", "cat", "dog", "dog", "cat", "dog", "dog"],
#     }
# )

#     categorical=["G", "H", "I"]
#     cat_vars = [X.columns.get_loc(var) for var in categorical]

#     cat_imputed_X,key_label_mapping = _integer_encoding(X,cat_vars=cat_vars)
#     print(cat_imputed_X)
    


#     print(key_label_mapping)

#     print(X)
    import pandas as pd
    import numpy as np

    data = {
    'Name': ['John', 'Jane', 'Mike', 'Sarah'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'London', 'Paris', 'Tokyo']
    }

    df = pd.DataFrame(data)

    # Create meshgrid of row and column indices
    row_index = df.index.values
    column_index = df.columns.values
    row_mesh, column_mesh = np.meshgrid(row_index, column_index)

    # Convert meshgrid to DataFrame
    mesh_df = pd.DataFrame({'Row': row_mesh.flatten(), 'Column': column_mesh.flatten()})

    print(mesh_df)

        