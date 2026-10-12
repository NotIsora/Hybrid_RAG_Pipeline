import json
import ast
import os

with open("notebooks/rag_colab_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")
all_ok = True
for idx, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    if cell_type == "code":
        code = "".join(cell["source"])
        lines = code.split("\n")
        clean_lines = []
        skip_pip = False
        for line in lines:
            s = line.strip()
            if s.startswith("!"):
                if s.endswith("\\"):
                    skip_pip = True
                continue
            if skip_pip:
                if not s.endswith("\\"):
                    skip_pip = False
                continue
            clean_lines.append(line)
        py_code = "\n".join(clean_lines)
        try:
            ast.parse(py_code)
            print(f"  [OK] Cell {idx} ({cell_type}) parsed successfully.")
        except Exception as e:
            print(f"  [FAIL] Cell {idx} syntax error: {e}")
            all_ok = False

if all_ok:
    print("\nALL NOTEBOOK CODE CELLS ARE 100% VALID PYTHON SYNTAX!")
