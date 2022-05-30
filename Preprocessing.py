import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import pickle
from Application_logging.logger_class import App_Logger


class Preprocessor:
    def __init__(self):
        try:
            self.logging = App_Logger()
            self.file_object =open("Logs/train_logs.txt",'a+')
            self.target = 'Petrol (USD)'
        except Exception as e:
            raise Exception(f"(__init__): Something went wrong on initiation process\n" + str(e))

    def readDataset(self,path):
        """
                This function helps to read the train dataset
        """
        self.logging.log(self.file_object, 'Entered the readDataset method of the Preprocessor class')
        try:
            data = pd.read_csv(path)
            self.logging.log(self.file_object, 'Reading data is a success')
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while reading the dataset\n"+str(e))

    def checkNullvalues(self,data):
        """
        This function helps to check missing values in the dataset
        """
        self.logging.log(self.file_object,'Entered the checkNullvalues method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts = data.isnull().sum()
            for i in self.null_counts:
                if i>0:
                    self.null_present = True
                    break
            self.logging.log(self.file_object,'Finding missing values is a success')
            return self.null_present
        except Exception as e:
            raise Exception(f"(checkNullvalues)-Something went wrong in checking null values\n"+str(e))

    def fillMissingValues(self,data):
        """
        This function helps to fill missing values using forward fill method
        """
        self.logging.log(self.file_object, 'Entered the fillMissingValues method of the Preprocessor class')
        try:
            data = data.fillna(method='ffill')
            if self.checkNullvalues(data) == False:
                self.logging.log(self.file_object, 'There are no null values, thus filling missing values is a success')
                return data
        except Exception as e:
            raise Exception(f"(fillMissingValues)-Something went wrong while filling null values\n" + str(e))

    def checkOutliers(self,data):
        """
        This function helps to check outliers in the dataset
        """
        self.logging.log(self.file_object, 'Entered the checkOutliers method of the Preprocessor class')
        self.outliers_present = False
        try:
            sorted_data = data[self.target].sort_values(ascending=True)
            quantile1, quantile3 = np.percentile(sorted_data, [25, 75])
            iqr = quantile3 - quantile1
            lower_bound = quantile1 - (1.5 * iqr)
            upper_bound = quantile3 + (1.5 * iqr)
            outliers = [i for i in sorted_data if i < lower_bound or i > upper_bound]
            if len(outliers) > 0:
                self.outliers_present = True
            self.logging.log(self.file_object, 'Finding outliers is a success')
            return self.outliers_present,outliers
        except Exception as e:
            raise Exception(f"(checkOutliers)-Something went wrong while checking Outliers\n" + str(e))

    def replacingOutliers(self,data,outliers):
        """
        This function helps to replace outliers with mean values of previous and next week data
        """
        self.logging.log(self.file_object, 'Entered the replacingOutliers method of the Preprocessor class')
        try:
            index = [data[self.target][data[self.target] == i].index.tolist() for i in outliers]
            outlier_index = [j for i in index for j in i]
            mean_values = [(data[self.target][i - 1] + data[self.target][i + 1]) / 2 for i in outlier_index]
            train = data.replace(outliers, mean_values)
            #val1 = (data[self.target][78]+data[self.target][80])/2
            #val2 = (data[self.target][141]+data[self.target][143])/2
            #train = data.replace(outliers,[val1,val2])
            outliers_present,outliers = self.checkOutliers(train)
            if outliers_present == False:
                self.logging.log(self.file_object,'Saving the plot of target data after replacing outliers is a success')
                self.logging.log(self.file_object, 'There are no outliers, thus replacing outliers is a success')
                return train
        except Exception as e:
            raise Exception(f"(replacingOutliers)-Something went wrong while replacing outliers\n" + str(e))

    def creatingNewFeatures(self,data):
        """
        This function helps to create new features in the dataset
        """
        self.logging.log(self.file_object, 'Entered the creatingNewFeatures method of the Preprocessor class')
        try:
            data['Date'] = pd.to_datetime(data['Date'])
            ## Extracting year,month,day,dayofyear,dayofweek,weekofyear to get more insights of data
            data['year'] = data['Date'].dt.year
            data['month'] = data['Date'].dt.month
            data['day'] = data['Date'].dt.day
            data['dayofyear'] = data['Date'].dt.dayofyear
            data['dayofweek'] = data['Date'].dt.dayofweek
            data['weekofyear'] = data['Date'].dt.weekofyear
            data = data.drop('Date', axis=1)
            self.logging.log(self.file_object, 'Creating New features is a success')
            return data
        except Exception as e:
            raise Exception(f"(creatingNewFeatures)-Something went wrong while creating new features\n" + str(e))

    def removeCorrFeatures(self,data):
        """
        This function helps to remove Multicollinearity in the dataset
        """
        self.logging.log(self.file_object, 'Entered the removeCorrFeatures method of the Preprocessor class')
        try:
            data_corr = data.corr()
            columns = np.full((data_corr.shape[0],), True, dtype=bool)
            for i in range(data_corr.shape[0]):
                for j in range(i + 1, data_corr.shape[0]):
                    if data_corr.iloc[i, j] >= 0.9:
                        if columns[j]:
                            columns[j] = False
            selected_columns = data.columns[columns]
            dataset_corr = data[selected_columns]
            return dataset_corr
        except Exception as e:
            raise Exception(f"(removeCorrFeatures)-Something went wrong while removing correlated features\n" + str(e))

    def trainModel(self,data):
        """
        This function helps to train the dataset using models
        """
        self.logging.log(self.file_object, 'Entered the trainModel method of the Preprocessor class')
        try:
            y = data['Petrol (USD)']
            X = data.drop(['Petrol (USD)'],axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            models = {
                'Linear Regression': LinearRegression(),
                'Linear Regression(Ridge)': Ridge(),
                'Linear Regression(lasso)': Lasso(),
                'Support vector Regression': SVR(),
                'Random Forest Regressor': RandomForestRegressor(),
                'XGBoost Regressor': XGBRegressor(),
                'Catboost': CatBoostRegressor(verbose=0),
                'Light Gradient boosting': LGBMRegressor()
            }
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                print(model_name, 'trained')
            self.logging.log(self.file_object, 'Training different models is a success')
            ## Selecting the best parameters using hyperparameter tuning
            rfrs_model = RandomForestRegressor(n_estimators=200,
                                               min_samples_split=2,
                                               min_samples_leaf=1,
                                               max_features='auto',
                                               max_depth=14,
                                               min_impurity_decrease=0.0000001,
                                               random_state=100)
            rfrs_model.fit(X_train, y_train)
            y_pred_rfrs = rfrs_model.predict(X_test)
            mape = np.mean(np.abs((y_test-y_pred_rfrs)/y_test))*100
            print("MAPE score after hyperparameter tuning:",mape)
            self.logging.log(self.file_object, 'Selecting the best parameters using hyperparameter tuning is a success')
            ## saving the model
            pickle.dump(rfrs_model, open('Best_model/modelForPrediction.sav', 'wb'))
            self.logging.log(self.file_object, 'Saving the best model')
        except Exception as e:
            raise Exception(f"(trainModel)-Something went wrong while training the model\n" + str(e))

    def predictingTestData(self,data,filename):
        """
        This function helps to predict the future data
        """
        self.logging.log(self.file_object, 'Entered the predictingTestData method of the Preprocessor class')
        try:
            with open("Best_model/rf_model_saved.sav",'rb') as f:
                model = pickle.load(f)
            forecast = model.predict(data)
            test = pd.read_csv('Testing_file/{}'.format(filename))
            test['Prediction'] = forecast
            test.to_csv('Forecast_result/forecast_result.csv', index=False)
            return test
        except Exception as e:
            raise Exception(f"(predictingTestData)-Something went wrong while predicting the test data\n" + str(e))

    def toDatetime(self,date):
        """
        This function helps to predict the future data
        """
        self.logging.log(self.file_object, 'Entered the toDatetime method of the Preprocessor class')
        try:
            date = pd.to_datetime(date)
            return date
        except Exception as e:
            raise Exception(f"(toDatetime)-Something went wrong while converting date to datetime\n" + str(e))

    def PreparingForPrediction(self,date):
        """
        This function helps to predict the future data
        """
        self.logging.log(self.file_object, 'Entered the PreparingForPrediction method of the Preprocessor class')
        try:
            date = pd.to_datetime(date)
            data = [[date.year,date.month,date.day,date.dayofweek]]
            return data
        except Exception as e:
            raise Exception(f"(PreparingForPrediction)-Something went wrong while preparing data fro prediction\n" + str(e))

    def predictingSingleDate(self,data):
        """
        This function helps to predict the petrol price for single date
        """
        self.logging.log(self.file_object, 'Entered the predictingSingleDate method of the Preprocessor class')
        try:
            with open("Best_model/modelForPrediction.sav", 'rb') as f:
                model = pickle.load(f)
            forecast = model.predict(data)
            return forecast
        except Exception as e:
            raise Exception(f"(predictingSingleDate)-Something went wrong while predicting petrol price for single date\n" + str(e))

    def saveFile(self,data,path):
        """
        This function saves data to a csv file
        """
        try:
            test_file = open(path,"w")
            test_file.write(str(data))
            test_file.close()
            test = pd.read_csv(path)
            return test
        except Exception as e:
            raise Exception(f"(saveFile) - Unable to save data to a csv file.\n" + str(e))




