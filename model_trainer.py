from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.model_x = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_y = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_w = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_h = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, data):
        features = ['plot_width', 'plot_depth', 'num_rooms', 'room_type_num']
        X = data[features]
        
        y_x = data['room_x']
        y_y = data['room_y']
        y_w = data['room_width']
        y_h = data['room_height']
        
        X_train, X_test, y_x_train, y_x_test = train_test_split(X, y_x, test_size=0.2, random_state=42)
        _, _, y_y_train, y_y_test = train_test_split(X, y_y, test_size=0.2, random_state=42)
        _, _, y_w_train, y_w_test = train_test_split(X, y_w, test_size=0.2, random_state=42)
        _, _, y_h_train, y_h_test = train_test_split(X, y_h, test_size=0.2, random_state=42)
        
        self.model_x.fit(X_train, y_x_train)
        self.model_y.fit(X_train, y_y_train)
        self.model_w.fit(X_train, y_w_train)
        self.model_h.fit(X_train, y_h_train)
        
        self.export_models()
        
        scores = {
            'x': self.model_x.score(X_test, y_x_test),
            'y': self.model_y.score(X_test, y_y_test),
            'width': self.model_w.score(X_test, y_w_test),
            'height': self.model_h.score(X_test, y_h_test)
        }
        
        return scores

    def export_models(self):
        """Export trained models to files"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model_x, 'models/model_x.joblib')
        joblib.dump(self.model_y, 'models/model_y.joblib')
        joblib.dump(self.model_w, 'models/model_w.joblib')
        joblib.dump(self.model_h, 'models/model_h.joblib')
        print("Models exported to 'models/' directory")

    def get_models(self):
        return {
            'x': self.model_x,
            'y': self.model_y,
            'width': self.model_w,
            'height': self.model_h
        }