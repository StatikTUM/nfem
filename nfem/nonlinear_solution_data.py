from mako.template import Template
import uuid
import sys

IS_NOTEBOOK = 'ipykernel' in sys.modules


class NonlinearSolutionInfo:
    def __init__(self, constraint, residual_norm, header, data):
        self.constraint = constraint
        self.header = header
        self.data = data
        self.residual_norm = residual_norm

    @property
    def iterations(self):
        return len(self.data)

    def show(self):
        if IS_NOTEBOOK:
            from IPython.display import display
            display(self)
        else:
            return f'Nonlinear solution converged after {len(data)} iterations'

    def _repr_html_(self):
        template = Template(TEMPLATE)

        return template.render(id=uuid.uuid4(), header=self.header, data=self.data)


TEMPLATE = '''
<style>
    .collapsible, .content {
        font-family: monospace;
        font-size: 14px;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    .collapsible {
        text-decoration: underline;
        cursor: pointer;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
    }

    .content {
        display: none;
        overflow: hidden;
        width: 100%;
    }
    
    tr, td {
        padding: 0 1em;
    }
</style>

<button type="button" class="collapsible collapsible-${id}">Nonlinear solution converged after ${len(data)} iterations</button>
<div class="content">
    <table>
    <tr>
        <th>k</th>
    % for entry in header:
        <th>${entry}</th>
    % endfor
    </tr>

    % for row in data:
    <tr>
        <td>${loop.index+1}</td>
    % for col in row:
        <td>${col}</td>
    % endfor
    </tr>
    % endfor
    </table>
</div>

<script>
    var coll = document.getElementsByClassName("collapsible-${id}");
    var i;

    for (i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
        content.style.display = "none";
        } else {
        content.style.display = "block";
        }
    });
    }
</script>
'''
