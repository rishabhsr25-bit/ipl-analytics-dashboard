from src.data_loader import IPLDataLoader
from src.predictor import IPLPredictor

print("Initializing Data Loader...")
loader = IPLDataLoader(matches_path="data/matches.csv", deliveries_path="data/deliveries.csv")

if loader.load_raw_data():
    print("Data loaded! Extracting Machine Learning features...")
    ml_data = loader.prepare_ml_data()
    
    print(f"Feature engineering complete. Dataset shape: {ml_data.shape}")
    
    predictor = IPLPredictor()
    accuracy = predictor.train(ml_data)
    print("Training complete! The model has been saved as 'src/model.pkl'.")
else:
    print("Error: Could not load CSV data.")