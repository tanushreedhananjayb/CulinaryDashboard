import nbformat
from nbconvert import PythonExporter

notebook_path = "ML_Tasks.ipynb"
py_module_path = "ML_Tasks.py"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_node = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    source_code, _ = exporter.from_notebook_node(nb_node)

with open(py_module_path, 'w', encoding='utf-8') as f:
    f.write(source_code)

print(f"Successfully converted: {notebook_path} ➡ {py_module_path}")
