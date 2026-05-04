import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import pickle


class MLR:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)

            # Encoding
            self.df['city'] = self.df['city'].map(
                {'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3, 'Redmond': 4,
                 'Maple Valley': 5, 'North Bend': 6, 'Lake Forest Park': 7, 'Sammamish': 8,
                 'Auburn': 9, 'Des Moines': 10, 'Bothell': 11, 'Federal Way': 12,
                 'Kirkland': 13, 'Issaquah': 14, 'Woodinville': 15, 'Normandy Park': 16,
                 'Fall City': 17, 'Renton': 18, 'Carnation': 19, 'Snoqualmie': 20,
                 'Duvall': 21, 'Burien': 22, 'Covington': 23, 'Inglewood-Finn Hill': 24,
                 'Kenmore': 25, 'Newcastle': 26, 'Mercer Island': 27, 'Black Diamond': 28,
                 'Ravensdale': 29, 'Clyde Hill': 30, 'Algona': 31, 'Skykomish': 32,
                 'Tukwila': 33, 'Vashon': 34, 'Yarrow Point': 35, 'SeaTac': 36,
                 'Medina': 37, 'Enumclaw': 39, 'Snoqualmie Pass': 40, 'Pacific': 41,
                 'Beaux Arts Village': 42, 'Preston': 43, 'Milton': 44}).astype(int)

            self.df['country'] = self.df['country'].map({'USA': 0}).astype(int)

            # Split
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Scaling (VERY IMPORTANT)
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

            print(f"Training size: {len(self.X_train)}")
            print(f"Testing size: {len(self.X_test)}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error Line {er_line.tb_lineno}: {er_type} - {er_msg}")

    def training(self):
        try:
            # Models
            self.lr = LinearRegression()
            self.ridge = Ridge(alpha=1.0)
            self.lasso = Lasso(alpha=0.1)

            # Train all
            self.lr.fit(self.X_train, self.y_train)
            self.ridge.fit(self.X_train, self.y_train)
            self.lasso.fit(self.X_train, self.y_train)

            print("\n===== TRAINING PERFORMANCE =====")

            for name, model in [("Linear", self.lr), ("Ridge", self.ridge), ("Lasso", self.lasso)]:
                pred = model.predict(self.X_train)
                r2 = r2_score(self.y_train, pred)
                rmse = np.sqrt(mean_squared_error(self.y_train, pred))
                print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.2f}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error Line {er_line.tb_lineno}: {er_type} - {er_msg}")

    def testing(self):
        try:
            print("\n===== TESTING PERFORMANCE =====")

            best_model = self.ridge   # you can change this

            pred = best_model.predict(self.X_test)
            r2 = r2_score(self.y_test, pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, pred))

            print(f"Test R2: {r2:.4f}")
            print(f"Test RMSE: {rmse:.2f}")

            self.best_model = best_model

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error Line {er_line.tb_lineno}: {er_type} - {er_msg}")

    def saving_model(self):
        try:
            # Save model + scaler
            with open("Model.pkl", "wb") as f:
                pickle.dump((self.best_model, self.scaler), f)

            print("\nModel saved successfully!")

            # Load and test
            with open("Model.pkl", "rb") as f:
                model, scaler = pickle.load(f)

            sample = np.array([[3,1.5,1340,7912,1.5,0,0,3,1340,0,1955,2005,1,0]])
            sample = scaler.transform(sample)

            prediction = model.predict(sample)
            print(f"Loaded Model Prediction: {prediction[0]}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error Line {er_line.tb_lineno}: {er_type} - {er_msg}")


if __name__ == "__main__":
    path = "data.csv"

    obj = MLR(path)
    obj.training()
    obj.testing()
    obj.saving_model()