from mako.template import Template
import uuid


class NonlinearSolutionInfo:
    def __init__(self, header, data):
        self.header = header
        self.data = data

    @property
    def iterations(self):
        return len(self.data)

    def _repr_html_(self):
        template = Template(TEMPLATE)

        return template.render(id=uuid.uuid4(), header=self.header, data=self.data)


TEMPLATE = '''
<style>
    .collapsible, .content {
    font-family: monospace;
    }

    .collapsible {
    color: #444;
    text-decoration: underline;
    cursor: pointer;
    margin: 0;
    padding: 0;
    width: 100%;
    border: none;
    text-align: left;
    outline: none;
    font-size: 14px;
    }

    .content {
    padding: 0 18px;
    display: none;
    overflow: hidden;
    }

    td {
    text-align: center;
    min-width: 100px;
    }
</style>

<button type="button" class="collapsible collapsible-${id}">Nonlinear solution converged in ${len(data)} iterations</button>
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
