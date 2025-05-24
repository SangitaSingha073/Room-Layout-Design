from data_generator import DataGenerator
from preprocessor import Preprocessor
from model_trainer import ModelTrainer
from layout_generator import LayoutGenerator
import joblib
import os

def main():
    
    os.makedirs('models', exist_ok=True)
    
    print("Generating synthetic data...")
    data_gen = DataGenerator()
    synthetic_data = data_gen.generate_synthetic_data(num_samples=10000)
    
    print("Preprocessing data...")
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess(synthetic_data)
    
    # Save the scaler
    joblib.dump(preprocessor.get_scaler(), 'models/scaler.joblib')
    
    print("Training model...")
    trainer = ModelTrainer()
    scores = trainer.train(processed_data)
    print(f"Model scores: {scores}")
    
    layout_gen = LayoutGenerator(trainer.get_models(), preprocessor.get_scaler())
    plot_width, plot_depth = 50, 40
    num_rooms = 4
    room_types = ['living_room', 'kitchen', 'bedroom', 'bathroom']
    
    layout = layout_gen.generate_layout(plot_width, plot_depth, num_rooms, room_types)
    layout_gen.visualize_layout(layout, plot_width, plot_depth)
    layout_gen.save_layout(layout, 'sample_layout.json')
    
    report = f"""
    Room Layout Generator Report
    ===========================
    Dataset Size: 10000 samples
    Model: Random Forest Regressor
    Training Scores:
    - X-coordinate: {scores['x']:.2f}
    - Y-coordinate: {scores['y']:.2f}
    - Width: {scores['width']:.2f}
    - Height: {scores['height']:.2f}
    
    Preprocessing:
    - Standard scaling applied to numerical features
    - Room types converted to numerical values
    
    Output:
    - 10000 JSON files in 'synthetic_layouts' folder
    - Sample layout saved as 'sample_layout.json' and 'generated_layout.png'
    - Models saved in 'models/' directory
    - Scaler saved as 'models/scaler.joblib'
    """
    
    with open('report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
