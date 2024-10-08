import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Create Performance Table
def create_performancetable():
    performance = pd.DataFrame(columns=['MAE', 'RMSE', 'r^2', 'MAPE', 'Training Time', 'Prediction Time'])
    return performance

def to_pkl(model, filename: str):
    """
    Save a model to a .pkl file.
    
    Parameters:
    model: The model object to be saved
    filename (str): The name of the file to save the model to (including .pkl extension)
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model successfully saved to {filename}")



# Linear Regression Model
class LinearRegression_model():
    def __init__ (self, X_train, y_train, X_test, y_test, model_name = "Linear Regression"):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.train_time = None
        self.predict_time = None
        self.performance_table = None

    def fit_predict(self):
        """
        Fit and Predict with model

        returns:
        model: fitted model
        y_pred: model predictions
        """
    
        print("Fitting and Predicting Linear Regression")

        start = time.time()
        self.model = LinearRegression().fit(self.X_train, self.y_train)
        end_train = time.time()

        y_pred = self.model.predict(self.X_test)
        end_pred = time.time()
        
        self.train_time = end_train - start
        self.predict_time = end_pred - start
        return self.model, y_pred
        

    def eval(self, y_pred):
        """
        Evalute Model Performance

        args: 
        y_pred: Predictions on Test data

        returns:
        metrics: dictonary containing metric values

        """
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)
        r2= r2_score(self.y_test, y_pred)
        MAPE = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
        print(f"MAPE: {MAPE}")
        print(f"time to train: {self.train_time:.2f} s")
        print(f"time to predict: {self.predict_time:.2f} s")

        metrics = {
            'MAE' : MAE,
            'RMSE' : RMSE,
            'R2' : r2,
            'MAPE' : MAPE,
            'Train Time' : self.train_time,
            'Predict Time' : self.predict_time
        }

        self.metrics = metrics

        return metrics

    def resid_plot(self, y_pred):
        """
        Plot Residuals

        args: 
        y_pred: Predictions on Test data
        """

        resid = self.y_test-y_pred
        plt.scatter(y_pred, resid)
        plt.title('Residual Plot (Linear Regression)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('reports/figures/models/linreg_resid.png', format='png', dpi=300)
        plt.show()



    def vs_plot(self, y_pred):
        """
        Plot actual vs predicted values

        args:
        y_pred: Predictions on Test data
        """

        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted (Linear Regression)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('reports/figures/models/linreg_vs.png', format='png', dpi=300)
        plt.show()

    def map_performance(self, df, model_name = None):
        if model_name is None:
            model_name = self.model_name

        if not hasattr(self, 'metrics'):
            raise ValueError("Need To Run eval() before running map_performance")
        
        df.loc[model_name] = [self.metrics['MAE'],
                              self.metrics['RMSE'],
                              self.metrics['R2'],
                              self.metrics['MAPE'],
                              self.metrics['Train Time'],
                              self.metrics['Predict Time']]
        df
# Ridge Regression Model       
class RidgeRegression_model():
    def __init__ (self, X_train, y_train, X_test, y_test, model_name = "Ridge Regression", alphas=[.001,.1,1,10,100], cv=10):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.alphas = alphas
        self.cv = cv
        self.model = None
        self.train_time = None
        self.predict_time = None
        self.performance_table = None


    def fit_predict(self):
        """
        Fit and Predict with model

        returns:
        model: fitted model
        y_pred: model predictions
        """
    
        print("Fitting and Predicting Ridge Regression")

        start = time.time()
        self.model = RidgeCV(alphas=self.alphas, cv=self.cv)


        self.model.fit(self.X_train,self.y_train)
        print(f"Alpha: {self.model.alpha_}")

        end_train = time.time()

        y_pred = self.model.predict(self.X_test)
        end_pred = time.time()
        
        self.train_time = end_train - start
        self.predict_time = end_pred - start
        return self.model, y_pred
        

    def eval(self, y_pred):
        """
        Evalute Model Performance

        args: 
        y_pred: Predictions on Test data

        returns:
        metrics: dictonary containing metric values

        """
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)
        r2= r2_score(self.y_test, y_pred)
        MAPE = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
        print(f"MAPE: {MAPE}")
        print(f"time to train: {self.train_time:.2f} s")
        print(f"time to predict: {self.predict_time:.2f} s")

        metrics = {
            'MAE' : MAE,
            'RMSE' : RMSE,
            'R2' : r2,
            'MAPE' : MAPE,
            'Train Time' : self.train_time,
            'Predict Time' : self.predict_time
        }
        self.metrics = metrics

        return metrics

    def resid_plot(self, y_pred):
        """
        Plot Residuals

        args: 
        y_pred: Predictions on Test data
        """

        resid = self.y_test-y_pred
        plt.scatter(y_pred, resid)
        plt.title('Residual Plot (Ridge Regression)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('reports/figures/models/ridge_resid.png', format='png', dpi=300)
        plt.show()

    def vs_plot(self, y_pred):
        """
        Plot actual vs predicted values

        args:
        y_pred: Predictions on Test data
        """

        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted (Ridge Regression)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('reports/figures/models/ridge_vs.png', format='png', dpi=300)
        plt.show()

    def map_performance(self, df, model_name = None):
        if model_name is None:
            model_name = self.model_name

        if not hasattr(self, 'metrics'):
            raise ValueError("Need To Run eval() before running map_performance")
        
        df.loc[model_name] = [self.metrics['MAE'],
                              self.metrics['RMSE'],
                              self.metrics['R2'],
                              self.metrics['MAPE'],
                              self.metrics['Train Time'],
                              self.metrics['Predict Time']]
        df
# Lasso Regression Model
class LassoRegression_model():
    def __init__ (self, X_train, y_train, X_test, y_test, model_name = "Lasso Regression", alphas=[.0001,.001,.1,1,10], cv=10, max_iter=1000000):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.alphas = alphas
        self.cv = cv
        self.max_iter = max_iter
        self.model = None
        self.train_time = None
        self.predict_time = None
        self.performance_table = None


    def fit_predict(self):
        """
        Fit and Predict with model

        returns:
        model: fitted model
        y_pred: model predictions
        """
    
        print("Fitting and Predicting Lasso Regression")

        start = time.time()
        self.model = LassoCV(alphas=self.alphas, cv=self.cv, max_iter=self.max_iter)

        
        self.model.fit(self.X_train,self.y_train)
        print(f"Alpha: {self.model.alpha_}")

        end_train = time.time()

        y_pred = self.model.predict(self.X_test)
        end_pred = time.time()
        
        self.train_time = end_train - start
        self.predict_time = end_pred - start
        return self.model, y_pred
        

    def eval(self, y_pred):
        """
        Evalute Model Performance

        args: 
        y_pred: Predictions on Test data

        returns:
        metrics: dictonary containing metric values

        """
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)
        r2= r2_score(self.y_test, y_pred)
        MAPE = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
        print(f"MAPE: {MAPE}")
        print(f"time to train: {self.train_time:.2f} s")
        print(f"time to predict: {self.predict_time:.2f} s")

        metrics = {
            'MAE' : MAE,
            'RMSE' : RMSE,
            'R2' : r2,
            'MAPE' : MAPE,
            'Train Time' : self.train_time,
            'Predict Time' : self.predict_time
        }
        self.metrics = metrics

        return metrics

    def resid_plot(self, y_pred):
        """
        Plot Residuals

        args: 
        y_pred: Predictions on Test data
        """

        resid = self.y_test-y_pred
        plt.scatter(y_pred, resid)
        plt.title('Residual Plot (Lasso Regression)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('reports/figures/models/lasso_resid.png', format='png', dpi=300)
        plt.show()

    def vs_plot(self, y_pred):
        """
        Plot actual vs predicted values

        args:
        y_pred: Predictions on Test data
        """

        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted (Lasso Regression)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('reports/figures/models/lasso_vs.png', format='png', dpi=300)
        plt.show()

    def map_performance(self, df, model_name = None):
        if model_name is None:
            model_name = self.model_name

        if not hasattr(self, 'metrics'):
            raise ValueError("Need To Run eval() before running map_performance")
        
        df.loc[model_name] = [self.metrics['MAE'],
                              self.metrics['RMSE'],
                              self.metrics['R2'],
                              self.metrics['MAPE'],
                              self.metrics['Train Time'],
                              self.metrics['Predict Time']]
        df
# Decision Tree Model
class DecisionTree_model():
    def __init__ (self, X_train, y_train, X_test, y_test, model_name = "Decision Tree"):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.train_time = None
        self.predict_time = None
        self.performance_table = None


    def fit_predict(self):
        """
        Fit and Predict with model

        returns:
        model: fitted model
        y_pred: model predictions
        """
    
        print("Fitting and Predicting Decision Tree Regressor")

        start = time.time()
        self.model = DecisionTreeRegressor().fit(self.X_train,self.y_train)
        end_train = time.time()

        y_pred = self.model.predict(self.X_test)
        end_pred = time.time()
        
        self.train_time = end_train - start
        self.predict_time = end_pred - start
        return self.model, y_pred
        

    def eval(self, y_pred):
        """
        Evalute Model Performance

        args: 
        y_pred: Predictions on Test data

        returns:
        metrics: dictonary containing metric values

        """
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)
        r2= r2_score(self.y_test, y_pred)
        MAPE = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
        print(f"MAPE: {MAPE}")
        print(f"time to train: {self.train_time:.2f} s")
        print(f"time to predict: {self.predict_time:.2f} s")

        metrics = {
            'MAE' : MAE,
            'RMSE' : RMSE,
            'R2' : r2,
            'MAPE' : MAPE,
            'Train Time' : self.train_time,
            'Predict Time' : self.predict_time
        }
        self.metrics = metrics

        return metrics

    def resid_plot(self, y_pred):
        """
        Plot Residuals

        args: 
        y_pred: Predictions on Test data
        """

        resid = self.y_test-y_pred
        plt.scatter(y_pred, resid)
        plt.title('Residual Plot (Decision Tree)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('reports/figures/models/decision_resid.png', format='png', dpi=300)
        plt.show()

    def vs_plot(self, y_pred):
        """
        Plot actual vs predicted values

        args:
        y_pred: Predictions on Test data
        """

        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted (Decision Tree)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('reports/figures/models/decision_vs.png', format='png', dpi=300)
        plt.show()

    def map_performance(self, df, model_name = None):
        if model_name is None:
            model_name = self.model_name

        if not hasattr(self, 'metrics'):
            raise ValueError("Need To Run eval() before running map_performance")
        
        df.loc[model_name] = [self.metrics['MAE'],
                              self.metrics['RMSE'],
                              self.metrics['R2'],
                              self.metrics['MAPE'],
                              self.metrics['Train Time'],
                              self.metrics['Predict Time']]
        df      
# Random Forest Model
class RandomForest_model():
    def __init__ (self, X_train, y_train, X_test, y_test, model_name = "Random Forest"):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.train_time = None
        self.predict_time = None
        self.performance_table = None


    def fit_predict(self):
        """
        Fit and Predict with model

        returns:
        model: fitted model
        y_pred: model predictions
        """
    
        print("Fitting and Predicting Decision Tree Regressor")

        start = time.time()
        self.model = RandomForestRegressor().fit(self.X_train,self.y_train)
        end_train = time.time()

        y_pred = self.model.predict(self.X_test)
        end_pred = time.time()
        
        self.train_time = end_train - start
        self.predict_time = end_pred - start
        return self.model, y_pred
        

    def eval(self, y_pred):
        """
        Evalute Model Performance

        args: 
        y_pred: Predictions on Test data

        returns:
        metrics: dictonary containing metric values

        """
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)
        r2= r2_score(self.y_test, y_pred)
        MAPE = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
        print(f"MAPE: {MAPE}")
        print(f"time to train: {self.train_time:.2f} s")
        print(f"time to predict: {self.predict_time:.2f} s")

        metrics = {
            'MAE' : MAE,
            'RMSE' : RMSE,
            'R2' : r2,
            'MAPE' : MAPE,
            'Train Time' : self.train_time,
            'Predict Time' : self.predict_time
        }
        self.metrics = metrics

        return metrics

    def resid_plot(self, y_pred):
        """
        Plot Residuals

        args: 
        y_pred: Predictions on Test data
        """

        resid = self.y_test-y_pred
        plt.scatter(y_pred, resid)
        plt.title('Residual Plot (Random Forest)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('reports/figures/models/forest_resid.png', format='png', dpi=300)
        plt.show()

    def vs_plot(self, y_pred):
        """
        Plot actual vs predicted values

        args:
        y_pred: Predictions on Test data
        """

        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted (Random Forest)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('reports/figures/models/forest_vs.png', format='png', dpi=300)
        plt.show()

    def map_performance(self, df, model_name = None):
        if model_name is None:
            model_name = self.model_name

        if not hasattr(self, 'metrics'):
            raise ValueError("Need To Run eval() before running map_performance")
        
        df.loc[model_name] = [self.metrics['MAE'],
                              self.metrics['RMSE'],
                              self.metrics['R2'],
                              self.metrics['MAPE'],
                              self.metrics['Train Time'],
                              self.metrics['Predict Time']]  
        df  
# Gradient Boosting Model
class GradientBoost_model():
    def __init__ (self, X_train, y_train, X_test, y_test, model_name = "Gradient Boosting"):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.train_time = None
        self.predict_time = None
        self.performance_table = None


    def fit_predict(self):
        """
        Fit and Predict with model

        returns:
        model: fitted model
        y_pred: model predictions
        """
    
        print("Fitting and Predicting Decision Tree Regressor")

        start = time.time()
        self.model = GradientBoostingRegressor().fit(self.X_train,self.y_train)
        end_train = time.time()

        y_pred = self.model.predict(self.X_test)
        end_pred = time.time()
        
        self.train_time = end_train - start
        self.predict_time = end_pred - start
        return self.model, y_pred
        

    def eval(self, y_pred):
        """
        Evalute Model Performance

        args: 
        y_pred: Predictions on Test data

        returns:
        metrics: dictonary containing metric values

        """
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)
        r2= r2_score(self.y_test, y_pred)
        MAPE = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
        print(f"MAPE: {MAPE}")
        print(f"time to train: {self.train_time:.2f} s")
        print(f"time to predict: {self.predict_time:.2f} s")

        metrics = {
            'MAE' : MAE,
            'RMSE' : RMSE,
            'R2' : r2,
            'MAPE' : MAPE,
            'Train Time' : self.train_time,
            'Predict Time' : self.predict_time
        }
        self.metrics = metrics

        return metrics

    def resid_plot(self, y_pred):
        """
        Plot Residuals

        args: 
        y_pred: Predictions on Test data
        """

        resid = self.y_test-y_pred
        plt.scatter(y_pred, resid)
        plt.title('Residual Plot (Gradient Boosting)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('reports/figures/models/gradient_resid.png', format='png', dpi=300)
        plt.show()

    def vs_plot(self, y_pred):
        """
        Plot actual vs predicted values

        args:
        y_pred: Predictions on Test data
        """

        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted (Gradient Boosting)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('reports/figures/models/gradient_vs.png', format='png', dpi=300)
        plt.show()

    def map_performance(self, df, model_name = None):
        if model_name is None:
            model_name = self.model_name

        if not hasattr(self, 'metrics'):
            raise ValueError("Need To Run eval() before running map_performance")
        
        df.loc[model_name] = [self.metrics['MAE'],
                              self.metrics['RMSE'],
                              self.metrics['R2'],
                              self.metrics['MAPE'],
                              self.metrics['Train Time'],
                              self.metrics['Predict Time']]
        df



"""
EX:
df = pd.read_csv('data/processed/fin_data.csv')

X = df.drop('t_cap',axis=1)
y = df['t_cap']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .3, random_state= 0)



model = LinearRegression_model(X_train, y_train, X_test, y_test)

# Fit the model and get predictions
_, y_pred = model.fit_predict()

# Evaluate the model
metrics = model.eval(y_pred)

model.resid_plot(y_pred)
model.vs_plot(y_pred)

# Now you can map the performance
model.map_performance(performance)
"""