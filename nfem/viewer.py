from IPython.display import display_html
from xml.dom.minidom import parse as parse_xml
import json
import pathlib
import html


ROOT = pathlib.Path(__file__).parent.resolve()


def load_html(name: str, data):
    path = f'{ROOT}/html/{name}.html'

    with open(path, 'r') as f:
        html = f.read()

    start_tag = '<script type="application/json">'
    end_tag = '</script>'

    start_index = html.find(start_tag) + len(start_tag)

    head = html[:start_index]
    tail = html[start_index:]

    end_index = html.find(end_tag)

    tail = html[end_index:]

    return head + json.dumps(data) + tail
