import json
import html
import os
from IPython.display import display_html


class Item:
    def __init__(self, data, id):
        self._data = data
        data['id'] = id
        data['geometry'] = []
        data['results'] = {}

    def add_result(self, key, value):
        results = self._data['results']

        results[key] = value

    def vec(self, v):
        return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}

    def set_label_location(self, ref, act):
        self._data['ref_label_location'] = self.vec(ref)
        self._data['act_label_location'] = self.vec(act)

    def add_point(self, location, color, layer):
        self._data['geometry'].append({
            'type': 'point',
            'location': self.vec(location),
            'layer': layer,
            'color': color,
        })

    def add_line(self, points, color, layer):
        self._data['geometry'].append({
            'type': 'line',
            'points': [self.vec(point) for point in points],
            'layer': layer,
            'color': color,
        })

    def add_arrow(self, location, direction, color, layer):
        self._data['geometry'].append({
            'type': 'arrow',
            'location': self.vec(location),
            'direction': self.vec(direction),
            'layer': layer,
            'color': color,
        })

    def add_support(self, location, direction, color, layer):
        self._data['geometry'].append({
            'type': 'support',
            'location': self.vec(location),
            'direction': direction,
            'layer': layer,
            'color': color,
        })

    def spring(self, *args, **kwargs):
        pass


class Canvas3D:
    def __init__(self, height=600):
        self.settings = dict()
        self.settings['timestep'] = 0

        self.data = dict()
        self.data['timesteps'] = []

    def _embed_js(self, template, filename):
        path = os.path.join(os.path.dirname(__file__), 'html', filename)
        with open(path, 'r', encoding='UTF-8') as file:
            content = file.read()
            template = template.replace(f'<script type="text/javascript" src="{filename}"></script>', f'<script type="text/javascript">{content}</script>')
        return template

    def html(self, height, model):
        content = self.raw_html(height, model)

        return f'<iframe seamless frameborder="0" allowfullscreen width="100%" height="{height}" srcdoc="{html.escape(content)}"></iframe>'

    def raw_html(self, height, model):
        timesteps_data = []

        for model in model.get_model_history():
            nodes_data = []

            for node in model.nodes:
                node_data = {}
                item = Item(node_data, node.id)
                node.draw(item)
                nodes_data.append(node_data)

            elements_data = []

            for element in model.elements:
                if element.__class__.__name__ == 'Spring':  # FIXME:
                    continue
                element_data = {}
                item = Item(element_data, element.id)
                element.draw(item)
                elements_data.append(element_data)

            timestep_data = {
                'nodes': nodes_data,
                'elements': elements_data,
            }

            timesteps_data.append(timestep_data)

        data = {
            'timesteps': timesteps_data
        }

        template_path = os.path.join(os.path.dirname(__file__), 'html', 'index.html')
        with open(template_path, 'r', encoding='UTF-8') as file:
            template = file.read()

        template = self._embed_js(template, 'index.js')

        content = template.replace("const data = {}", "const data = " + json.dumps(data))
        
        return content

    def show(self, height, model):
        display_html(self.html(height, model), raw=True)
