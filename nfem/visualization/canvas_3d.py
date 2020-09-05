import json
import numpy as np
import html
from IPython.display import display, HTML
from nfem.visualization.canvas_3d_html import TEMPLATE


class Canvas3D:
    def __init__(self, height=600):
        self.data = dict()
        self.data['height'] = height
        self.data['frame'] = 0
        self.data['frames'] = []

    def next_frame(self):
        self.data['frames'].append({"items": []})

    @property
    def frame(self):
        return self.data['frame']

    @frame.setter
    def frame(self, value):
        self.data['frame'] = value

    def line(self, a, b, color='red', layer=0):
        items = self.data['frames'][-1]['items']

        a = np.asarray(a)
        b = np.asarray(b)

        items.append({
            "type": "line",
            "points": [
                {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
                {"x": float(b[0]), "y": float(b[1]), "z": float(b[2])},
            ],
            "color": color,
            "layer": layer,
        })

    def polyline(self, points, color='red', layer=0):
        items = self.data['frames'][-1]['items']

        points = np.asarray(points)

        items.append({
            "type": "line",
            "points": [{"x": point[0], "y": point[1], "z": point[2]} for point in points],
            "color": color,
            "layer": layer,
        })

    def point(self, location, color='red', layer=0):
        items = self.data['frames'][-1]['items']

        location = np.asarray(location)

        items.append({
            "type": "point",
            "location": {"x": float(location[0]), "y": float(location[1]), "z": float(location[2])},
            "color": color,
            "layer": layer,
        })

    def arrow(self, location, direction, color='red', layer=0):
        items = self.data['frames'][-1]['items']

        location = np.asarray(location)
        direction = np.asarray(direction)

        items.append({
            "type": "arrow",
            "location": {"x": float(location[0]), "y": float(location[1]), "z": float(location[2])},
            "direction": {"x": float(direction[0]), "y": float(direction[1]), "z": float(direction[2])},
            "color": color,
            "layer": layer,
        })

    def text(self, text, location, color='red', layer=0):
        items = self.data['frames'][-1]['items']

        location = np.asarray(location)

        items.append({
            "type": "text",
            "text": text,
            "location": {"x": float(location[0]), "y": float(location[1]), "z": float(location[2])},
            "color": color,
            "layer": layer,
        })

    def support(self, direction, location, color='red', layer=0):
        items = self.data['frames'][-1]['items']

        location = np.asarray(location)

        items.append({
            "type": "support",
            "direction": direction,
            "location": {"x": float(location[0]), "y": float(location[1]), "z": float(location[2])},
            "color": color,
            "layer": layer,
        })

    def show(self, height):
        content = TEMPLATE.replace("{{data}}", json.dumps(self.data))
        element = HTML(f'<iframe seamless frameborder="0" allowfullscreen width="100%" height="{height+100}" srcdoc="{html.escape(content)}"</iframe>')
        display(element)
