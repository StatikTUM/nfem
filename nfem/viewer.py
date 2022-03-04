from IPython.display import display_html
from xml.dom.minidom import parse as parse_xml
import json
import pathlib
import html


ROOT = pathlib.Path(__file__).parent.resolve()


def load_html(name: str, data):
    path = f'{ROOT}/html/{name}.html'

    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()

    start_tag = '<script type="application/json">'
    end_tag = '</script>'

    start_index = html.find(start_tag) + len(start_tag)

    head = html[:start_index]
    tail = html[start_index:]

    end_index = html.find(end_tag)

    tail = html[end_index:]

    return head + json.dumps(data) + tail


def show_html(raw_html, height=400, iframe=False):
    import sys
    if 'ipykernel' in sys.modules:
        from html import escape
        from IPython.display import display_html

        if iframe:
            display_html(f'<iframe seamless frameborder="0" allowfullscreen width="100%" height="{height}" srcdoc="{escape(raw_html)}"></iframe>', raw=True)
        else:
            display_html(raw_html, raw=True)
    else:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWidgets import QApplication

        import sys

        app = QApplication(sys.argv + ['--disable-logging'])

        view = QWebEngineView()
        view.setWindowTitle('nfem')
        view.setHtml(raw_html)
        view.show()

        sys.exit(app.exec())
