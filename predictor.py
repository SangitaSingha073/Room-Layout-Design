import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import os

class RoomLayoutPredictor:
    def __init__(self, model_dir='models'):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please train the model first.")
        
        try:
            self.model_x = joblib.load(os.path.join(model_dir, 'model_x.joblib'))
            self.model_y = joblib.load(os.path.join(model_dir, 'model_y.joblib'))
            self.model_w = joblib.load(os.path.join(model_dir, 'model_w.joblib'))
            self.model_h = joblib.load(os.path.join(model_dir, 'model_h.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing model file: {e}. Ensure all model files are in '{model_dir}'.")
        
        self.room_type_mapping = {'bedroom': 0, 'bathroom': 1, 'kitchen': 2, 'living_room': 3}
        self.valid_room_types = list(self.room_type_mapping.keys())

    def get_user_input(self):
        print("\n=== Room Layout Specification ===")
        
        while True:
            try:
                plot_width = float(input("Enter plot width (20-100): "))
                if 20 <= plot_width <= 100:
                    break
                print("Plot width must be between 20 and 100")
            except ValueError:
                print("Please enter a valid number")
                
        while True:
            try:
                plot_depth = float(input("Enter plot depth (20-100): "))
                if 20 <= plot_depth <= 100:
                    break
                print("Plot depth must be between 20 and 100")
            except ValueError:
                print("Please enter a valid number")
                
        while True:
            try:
                num_rooms = int(input("Enter number of rooms (2-8): "))
                if 2 <= num_rooms <= 8:
                    break
                print("Number of rooms must be between 2 and 8")
            except ValueError:
                print("Please enter a valid integer")
        
        print(f"\nAvailable room types: {', '.join(self.valid_room_types)}")
        room_types = []
        for i in range(num_rooms):
            while True:
                room_type = input(f"Enter type for room {i+1} (or press Enter for default 'bedroom'): ").lower()
                if room_type == "":
                    room_types.append('bedroom')
                    print(f"Room {i+1} set to default: bedroom")
                    break
                elif room_type in self.room_type_mapping:
                    room_types.append(room_type)
                    print(f"Room {i+1} set to: {room_type}")
                    break
                print(f"Invalid room type. Please use: {', '.join(self.valid_room_types)}")
        
        return plot_width, plot_depth, num_rooms, room_types

    def check_overlap(self, room1, room2):
        """Check if two rooms overlap."""
        x1, y1, w1, h1 = room1['x'], room1['y'], room1['width'], room1['height']
        x2, y2, w2, h2 = room2['x'], room2['y'], room2['width'], room2['height']
        
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)

    def adjust_layout(self, layout, plot_width, plot_depth):
        """Adjust room positions to prevent overlaps."""
        adjusted_layout = layout.copy()
        
        for i in range(len(adjusted_layout)):
            for j in range(i + 1, len(adjusted_layout)):
                while self.check_overlap(adjusted_layout[i], adjusted_layout[j]):
                    # Shift room j to the right
                    adjusted_layout[j]['x'] += adjusted_layout[i]['width'] * 0.5
                    # If it exceeds plot width, move it down and reset x
                    if adjusted_layout[j]['x'] + adjusted_layout[j]['width'] > plot_width:
                        adjusted_layout[j]['x'] = 0
                        adjusted_layout[j]['y'] += adjusted_layout[i]['height'] * 0.5
                    
                    # Ensure room stays within plot boundaries
                    adjusted_layout[j]['x'] = max(0, min(adjusted_layout[j]['x'], plot_width - adjusted_layout[j]['width']))
                    adjusted_layout[j]['y'] = max(0, min(adjusted_layout[j]['y'], plot_depth - adjusted_layout[j]['height']))
        
        return adjusted_layout

    def predict_layout(self, plot_width, plot_depth, num_rooms, room_types):
        """Predict room layout and adjust to prevent overlaps."""
        input_data = pd.DataFrame({
            'plot_width': [plot_width] * num_rooms,
            'plot_depth': [plot_depth] * num_rooms,
            'num_rooms': [num_rooms] * num_rooms,
            'room_type_num': [self.room_type_mapping[t] for t in room_types]
        })
        
        features_to_scale = ['plot_width', 'plot_depth', 'num_rooms']
        try:
            input_scaled = self.scaler.transform(input_data[features_to_scale])
            input_data[features_to_scale] = input_scaled
        except ValueError as e:
            raise ValueError(f"Scaler transformation failed: {e}. Ensure model was trained correctly.")
        
        x_coords = self.model_x.predict(input_data)
        y_coords = self.model_y.predict(input_data)
        widths = self.model_w.predict(input_data)
        heights = self.model_h.predict(input_data)
        
        layout = []
        for i in range(num_rooms):
            room = {
                'x': max(0, min(x_coords[i], plot_width - widths[i])),
                'y': max(0, min(y_coords[i], plot_depth - heights[i])),
                'width': min(widths[i], plot_width),
                'height': min(heights[i], plot_depth),
                'type': room_types[i]
            }
            layout.append(room)
        
        # Adjust layout to prevent overlaps
        adjusted_layout = self.adjust_layout(layout, plot_width, plot_depth)
        return adjusted_layout

    def visualize_layout(self, layout, plot_width, plot_depth, filename='predicted_layout.png'):
        """Visualize the predicted layout."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, plot_width)
        ax.set_ylim(0, plot_depth)
        
        colors = {'bedroom': 'blue', 'bathroom': 'green', 'kitchen': 'red', 'living_room': 'yellow'}
        
        for room in layout:
            rect = plt.Rectangle((room['x'], room['y']), room['width'], room['height'],
                               alpha=0.5, facecolor=colors.get(room['type'], 'gray'),
                               edgecolor='black')
            ax.add_patch(rect)
            ax.text(room['x'] + room['width']/2, room['y'] + room['height']/2,
                   room['type'], ha='center', va='center')
        
        plt.title(f'Predicted Room Layout ({plot_width}x{plot_depth})')
        plt.savefig(filename)
        plt.close()
        print(f"Layout visualization saved as '{filename}'")

    def save_layout(self, layout, filename='predicted_layout.json'):
        """Save the predicted layout to JSON."""
        with open(filename, 'w') as f:
            json.dump(layout, f, indent=2)
        print(f"Layout data saved as '{filename}'")

def main():
    try:
        predictor = RoomLayoutPredictor()
        
        plot_width, plot_depth, num_rooms, room_types = predictor.get_user_input()
        
        print("\nGenerating layout prediction...")
        layout = predictor.predict_layout(plot_width, plot_depth, num_rooms, room_types)
        
        predictor.visualize_layout(layout, plot_width, plot_depth)
        predictor.save_layout(layout)
        
        print("\nPrediction completed successfully!")
        print(f"Layout details: {json.dumps(layout, indent=2)}")
        
    except FileNotFoundError as e:
        print(e)
        print("Please run main.py first to train and export the models.")
    except ValueError as e:
        print(f"Prediction error: {e}")
        print("Please check your input or model compatibility.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your input or model files and try again.")

if __name__ == "__main__":
    main()