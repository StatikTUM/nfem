from IPython.display import display_html
from xml.dom.minidom import parse as parse_xml
import json
import pathlib
import html


ROOT = pathlib.Path(__file__).parent.resolve()


class Viewer:
    def html(self, models, height):
        timesteps = []

        for model in models:
            timestep = {}

            timestep['name'] = model.name or ''

            timestep['objects'] = objects = []

            for element in model.elements:
                element.draw(objects)

            for node in model.nodes:
                node.draw(objects)

            timesteps.append(timestep)

        data = dict(
            settings=dict(
                height=height,
            ),
            timesteps=timesteps,
        )

        return _load_with_data(f'{ROOT}/html/viewer.html', data)

    def frame(self, models, height):
        html_content = self.html(models, height)

        return _iframe(html_content, height)


def load_html(name: str, data):
    return _load_with_data(f'{ROOT}/html/{name}.html', data)


def load_iframe(name: str, data):
    content = load_html(name, data)

    return f'<iframe seamless frameborder="0" allowfullscreen width="100%" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+\'px\';" srcdoc="{html.escape(content)}"></iframe>'


def _iframe(content: str, height: int):
    return f'<iframe seamless frameborder="0" allowfullscreen width="100%" height="{height}" srcdoc="{html.escape(content)}"></iframe>'


def _load_with_data(path, data):
    with open(path, 'r') as f:
        html = f.read()

    start_tag = '<script id="nfem-data" type="application/json">'
    end_tag = '</script>'

    start_index = html.find(start_tag) + len(start_tag)

    head = html[:start_index]
    tail = html[start_index:]

    end_index = html.find(end_tag)

    tail = html[end_index:]

    return head + json.dumps(data) + tail
