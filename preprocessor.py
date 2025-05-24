import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.room_type_mapping = {'bedroom': 0, 'bathroom': 1, 'kitchen': 2, 'living_room': 3}

    def preprocess(self, df):
        """Preprocess the synthetic data"""
        df_processed = df.copy()
        df_processed['room_type_num'] = df_processed['room_type'].map(self.room_type_mapping)
        
        # Only scale input features
        features_to_scale = ['plot_width', 'plot_depth', 'num_rooms']
        scaled_features = self.scaler.fit_transform(df_processed[features_to_scale])
        df_processed[features_to_scale] = scaled_features
        
        return df_processed

    def get_scaler(self):
        return self.scaler