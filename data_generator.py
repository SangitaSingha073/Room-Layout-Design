import numpy as np
import pandas as pd
import json
from pathlib import Path

class DataGenerator:
    def __init__(self, output_dir="synthetic_layouts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_synthetic_data(self, num_samples=10000):
        data = {
            'plot_width': [],
            'plot_depth': [],
            'num_rooms': [],
            'room_x': [],
            'room_y': [],
            'room_width': [],
            'room_height': [],
            'room_type': []
        }
        
        room_types = ['bedroom', 'bathroom', 'kitchen', 'living_room']
        
        for i in range(num_samples):
            plot_width = np.random.uniform(20, 100)
            plot_depth = np.random.uniform(20, 100)
            n_rooms = np.random.randint(2, 8)
            
            for _ in range(n_rooms):
                width = min(np.random.uniform(8, plot_width/2), plot_width)
                height = min(np.random.uniform(8, plot_depth/2), plot_depth)
                x = np.random.uniform(0, plot_width - width)
                y = np.random.uniform(0, plot_depth - height)
                
                data['plot_width'].append(plot_width)
                data['plot_depth'].append(plot_depth)
                data['num_rooms'].append(n_rooms)
                data['room_x'].append(x)
                data['room_y'].append(y)
                data['room_width'].append(width)
                data['room_height'].append(height)
                data['room_type'].append(np.random.choice(room_types))
                
            layout = {
                'plot_width': plot_width,
                'plot_depth': plot_depth,
                'rooms': [
                    {
                        'x': data['room_x'][-j-1],
                        'y': data['room_y'][-j-1],
                        'width': data['room_width'][-j-1],
                        'height': data['room_height'][-j-1],
                        'type': data['room_type'][-j-1]
                    } for j in range(n_rooms)
                ]
            }
            with open(self.output_dir / f'layout_{i}.json', 'w') as f:
                json.dump(layout, f)
        
        return pd.DataFrame(data)