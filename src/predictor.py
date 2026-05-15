import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle
import os

class IPLPredictor:
    def __init__(self, model_path="src/model.pkl"):
        self.model_path = model_path
        self.pipeline = None

    def train(self, ml_df: pd.DataFrame):
        """Trains the XGBoost pipeline and saves it to disk."""
        print("Preparing data for training...")
        
        # 1. Split Features (X) and Target (y)
        X = ml_df.drop('result', axis=1)
        y = ml_df['result']

        # 2. Train-Test Split (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Create a preprocessing step for categorical variables (Teams & Venues)
        trf = ColumnTransformer([
            ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), 
             ['batting_team', 'bowling_team', 'venue'])
        ], remainder='passthrough')

        # 4. Build the Pipeline (Preprocessor -> XGBoost Model)
        self.pipeline = Pipeline(steps=[
            ('step1', trf),
            ('step2', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ])

        # 5. Train the Model
        print("Training XGBoost Model (This might take 10-30 seconds)...")
        self.pipeline.fit(X_train, y_train)

        # 6. Check Accuracy
        accuracy = self.pipeline.score(X_test, y_test)
        print(f"Model Trained Successfully! Accuracy: {accuracy * 100:.2f}%")

        # 7. Save model to disk
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
            
        return accuracy

    def load_model(self):
        """Loads the trained model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            return True
        return False

    def predict_probability(self, match_state_df: pd.DataFrame):
        """Takes a 1-row DataFrame and returns the win probability."""
        if self.pipeline is None:
            self.load_model()
            
        prob = self.pipeline.predict_proba(match_state_df)[0]
        # prob[0] is bowling team win chance, prob[1] is batting team win chance
        return {
            "batting_team_win_prob": round(prob[1] * 100, 1),
            "bowling_team_win_prob": round(prob[0] * 100, 1)
        }