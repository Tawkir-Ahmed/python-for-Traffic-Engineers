#Book: Hands on Machine learning
#https://github.com/tuitet/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-3rd-Edition/tree/main
#Part 1: Chapter 1
#Example 1-1: Training and running a linear model suing Scikit-Learn

import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Download and prepeare the data

url= "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/gdp_per_capita.csv"
#lifesat= pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/gdp_per_capita.csv')
lifesat= pd.read_csv(r'C:\Users\Tawkir\Desktop\Python Ex\lifesat.csv') # encoding='ISO-8859-1', 'gb18030', 'utf-8', 'gbk'
lifesat.head(3)
X = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values

#visualize the data
lifesat.plot(kind='scatter', grid=True,
             x='GDP per capita (USD)', y='Life satisfaction')
plt.axis([23_500, 62_500, 4, 9])
plt.show()

#select a linaer model
model = LinearRegression()

#Train the model
model.fit(X, y)

#Make a prediction for Cyprus
X_new= [[37_655.2]] # Cyprup' GDP per capital in 2020
print(model.predict(X_new))

#k-nearest neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)

#Chapter 2: End to end machine learning project
# Import data from online
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

'''
def load_housing_data():
    tarball_path = Path('datasets/housing.tgz')
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path='datasets')
    return pd.read_csv(Path('datasets/housing/housing.csv'))

housing = load_housing_data() # the system is not working, then download the datafile
'''


housing = pd.read_csv(r'C:\Users\Tawkir\Desktop\Python Ex\housing.csv')
housing.head()

housing.info()

housing['ocean_proximity'].value_counts()
housing.describe() #summary of each numerical attribute

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(12, 8)) # plot histogram of all numerical value
plt.show()

import numpy as np

#customize training and testing data without using train_test_split() function
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(housing, 0.2)
len(train_set)
len(test_set)

from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an index column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, 'index')

housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, 'id')

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# create dataframe
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing['income_cat'].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel('Income category')
plt.ylabel('Number of districts')
plt.show()

# data splits
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing['income_cat']):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing['income_cat'], random_state=42)

strat_test_set['income_cat'].value_counts() / len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind='scatter', x='longitude', y='latitude', grid=True)
plt.show()

#61
housing.plot(kind='scatter', x='longitude', y='latitude', grid=True, alpha=0.2)
plt.show() #make less and more severe plot

housing.plot(kind='scatter', x='longitude', y='latitude', grid=True,
             s=housing['population']/100, label= 'population',
             c='median_house_value', cmap='jet', colorbar=True,
             legend=True, sharex=False, figsize=(10,7))
plt.show() #color plot more interesting

#Correlation analysis
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False) # correlation housing with all var

from pandas.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show() # plot the correlation

housing.plot(kind='scatter', x='median_income', y='median_house_value',
             alpha=0.1, grid=True)
plt.show() #scatetr plot

# data modification
housing['rooms_per_house'] = housing['total_rooms'] / housing['households']
housing['bedrooms_ratio'] = housing['total_bedrooms'] / housing['total_rooms']
housing['people_per_house'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# Prepare the data for machine learning algorithm
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# Clean the Data
housing.dropna(subset=['total_bedrooms'], inplace=True) # option 1
housing.drop('total_bedrooms', axis=1) # option 2
median = housing['total_bedrooms'].median() #option 3
housing['total_bedrooms'].fillna(median, inplace= True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns= housing_num.columns, index=housing_num.index)

# Handling text and categrical attributes pg. 71
housing_cat = housing[['ocean_proximity']]
housing_cat.head(8)

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded =  ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8] #need to see the problem
ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_
df_test = pd.DataFrame({'ocean_proximity': ['INLAND', 'NEAR BAY']})
pd.get_dummies(df_test)
cat_encoder.transform(df_test)
df_test_unknown = pd.DataFrame({'ocean_proximity': ['<2H OCEAN', 'ISLAND']})
pd.get_dummies(df_test_unknown)
cat_encoder.handle_unknown = 'ignore'
cat_encoder.transform(df_test_unknown)
cat_encoder.feature_names_in_
cat_encoder.get_feature_names_out()
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns = cat_encoder.get_feature_names_out(),
                         index = df_test_unknown.index)

# Feature scaling and transformation
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[['housing_median_age']], [[35]], gamma=0.1)

from sklearn.linear_model import LinearRegression
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

print(housing[['median_income']].shape)
print(scaled_labels.shape)

# Assuming scaled_labels is a NumPy array or a pandas DataFrame
import numpy as np
# Reduce the number of samples in scaled_labels to match the shape of housing[['median_income']]
scaled_labels = scaled_labels[:16344]
# Now, both housing[['median_income']] and scaled_labels have the same shape
print(housing[['median_income']].shape)
print(scaled_labels.shape)


model = LinearRegression()
model.fit(housing[['median_income']], scaled_labels)
some_new_data = housing[['median_income']].iloc[:5] # pretend this is new data

scaled_predictions = model.predict(some_new_data) # error
predictions = target_scaler.inverse_transform(scaled_predictions)

from sklearn.compose import TransformedTargetRegressor
####################chekc in gpt
model = TransformedTargetRegressor(LinearRegression(),
                                   transformer = StandardScaler())
model.fit(housing[['median_income']], housing_labels)
predictions = model.predict(some_new_data)

# custom transformers
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func= np.exp)
log_pop = log_transformer.transform(housing[['population']])

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args= dict(Y=[[35.]], gamma=0.1))

age_simil_35 = rbf_transformer.transform(housing[['housing_median_age']])

sf_coords= 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[['latitude', 'longitude']])

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True): #no *args or **kwargs!
        self.with_mean = with_mean
    
    def fit(self, X, y=None): # y is required even though we don't use it
        X = check_array(X) # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1] # every estimatro stores this in fit()
        return self # always return self!
    
    def transform(self, X):
        check_is_fitted(self) # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight= sample_weight)
        return self # alsways return self!
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f'Cluster {i} similarity' for i in range(self.n_clusters)]
    

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[['latitude', 'longitude']],
                                           sample_weight= housing_labels)

similarities[:3].round(2)
#page=83

# Trsnformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),  # Corrected 'strantegy' to 'strategy'
    ('standardize', StandardScaler()),
])

from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)

from sklearn.compose import ColumnTransformer
num_attribs = ['longitude', 'latitude', 'housing_meadian_age', 'total_rooms',
               'total_bedrooms', 'population', 'households', 'median_income']
cat_attribs = ['ocean_proximity']

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include= np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transforemr, feature_names_in):
    return ['ratio'] # featur names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out= ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'),
                                     StandardScaler())

preprocessing = ColumnTransformer([
    ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']), # page: 87
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population',
                           'households', 'median_income']),
    ('geo', cluster_simil, ['latitude', 'longitude']),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object))],
remainder=default_num_pipeline) # one column remaining: housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape
preprocessing.get_feature_names_out()

# Select and Train a Model

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

print(housing.shape)
print(housing_labels.shape)

housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2) # -2 = rounded to the nearest hundred
housing_labels.iloc[:5].values
housing_labels.iloc[:5].values

from sklearn.metrics import mean_squared_error
lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
lin_rmse

# Decission Tree model
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
tree_rmse

# Better evaluation using cross-validation
from sklearn.model_selection import cross_val_score
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring='neg_root_mean_squared_error', cv=10) # iteration

pd.Series(tree_rmses).describe()

# RandomForest

from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring='neg_root_mean_squared_error', cv=10)
pd.Series(forest_rmses).describe()

#Fine-Tune Model
#Grid Search: fiddle with the hyperparameters manually until to find a best combination
#search best combination
from sklearn.model_selection import GridSearchCV
full_pipeline = ([
    ('preprocessing', preprocessing),
    ('random_forest', RandomForestRegressor(random_state=42)),
    ])
param_grid = [ 
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},   # 3*3=9 combination
     {'preprocessing__geo__n_clusters': [10, 15],  # 2*3=6 combination
      'random_forest__max_features': [6, 8, 10]}, # in total= 9+6= 15 explore
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)

grid_search.best_params_
#page = 93
import sklearn
print(sklearn.__version__)

from sklearn.model_selection import GridSearchCV

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by='mean_test_score', ascending=False, inplace=True)
cv_res.head()

''' the above code is not working need to try later '''

# Randomize Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)

# Ensemble Methods
# Analyzing the best models and their errors
final_model = rnd_search.best_estimator_  # includes proeprocessing
feature_importances = final_model['random_forest'].feature_importances_
feature_importances.round(2)
sorted(zip(feature_importances,
           final_model['preprocessing'].get_geature_names_out()),
           reverse=True)

'''page 93 to 97 not working need to try later'''

# Launchh, Monitor, and Maintain Your System
import joblib
joblib.dump(final_model, 'my_california_housing_model.pkl')


'''Ch: 2 last part does not match: need to see later'''
