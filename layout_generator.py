import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class LayoutGenerator:
    def __init__(self, models, scaler):
        self.models = models
        self.scaler = scaler
        self.room_type_mapping = {'bedroom': 0, 'bathroom': 1, 'kitchen': 2, 'living_room': 3}

    def check_overlap(self, room1, room2):
        x1, y1, w1, h1 = room1['x'], room1['y'], room1['width'], room1['height']
        x2, y2, w2, h2 = room2['x'], room2['y'], room2['width'], room2['height']
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)

    def adjust_layout(self, layout, plot_width, plot_depth):
        adjusted_layout = layout.copy()
        for i in range(len(adjusted_layout)):
            for j in range(i + 1, len(adjusted_layout)):
                while self.check_overlap(adjusted_layout[i], adjusted_layout[j]):
                    adjusted_layout[j]['x'] += adjusted_layout[i]['width'] * 0.5
                    if adjusted_layout[j]['x'] + adjusted_layout[j]['width'] > plot_width:
                        adjusted_layout[j]['x'] = 0
                        adjusted_layout[j]['y'] += adjusted_layout[i]['height'] * 0.5
                    adjusted_layout[j]['x'] = max(0, min(adjusted_layout[j]['x'], plot_width - adjusted_layout[j]['width']))
                    adjusted_layout[j]['y'] = max(0, min(adjusted_layout[j]['y'], plot_depth - adjusted_layout[j]['height']))
        return adjusted_layout

    def generate_layout(self, plot_width, plot_depth, num_rooms, room_types=None):
        if room_types is None:
            room_types = ['bedroom'] * num_rooms
            
        input_data = pd.DataFrame({
            'plot_width': [plot_width] * num_rooms,
            'plot_depth': [plot_depth] * num_rooms,
            'num_rooms': [num_rooms] * num_rooms,
            'room_type_num': [self.room_type_mapping[t] for t in room_types]
        })
        
        features_to_scale = ['plot_width', 'plot_depth', 'num_rooms']
        input_scaled = self.scaler.transform(input_data[features_to_scale])
        input_data[features_to_scale] = input_scaled
        
        x_coords = self.models['x'].predict(input_data)
        y_coords = self.models['y'].predict(input_data)
        widths = self.models['width'].predict(input_data)
        heights = self.models['height'].predict(input_data)
        
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
        
        return self.adjust_layout(layout, plot_width, plot_depth)

    def visualize_layout(self, layout, plot_width, plot_depth, filename='generated_layout.png'):
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
        
        plt.title(f'Generated Room Layout ({plot_width}x{plot_depth})')
        plt.savefig(filename)
        plt.close()

    def save_layout(self, layout, filename='layout.json'):
        with open(filename, 'w') as f:
            json.dump(layout, f, indent=2)