import os
import nbformat


KERNEL_NAME = "st-agent"
RESULTS_DIR = "results/"


def save_as_notebook(code: str, plan: dict, results_dir: str = RESULTS_DIR) -> str:
    """
    Convert a Python code string + plan into a .ipynb and save to results_dir.
    Returns the path to the written notebook.
    """
    os.makedirs(results_dir, exist_ok=True)

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": KERNEL_NAME,
        "language": "python",
        "name": KERNEL_NAME,
    }
    nb.metadata["language_info"] = {"name": "python"}

    # Markdown header cell with the plan goal and steps
    md_lines = [f"# Analysis: {plan.get('goal', '')}\n"]
    for step in plan.get("steps", []):
        fig = f" → `{step['figure_filename']}`" if step.get("produces_figure") else ""
        md_lines.append(f"- **[{step['step_id']}]** {step['description']}{fig}")
    nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(md_lines)))

    # Code cells: split on blank lines (paragraph structure)
    code = code.replace("\r\n", "\n")
    blocks = [b.strip() for b in code.split("\n\n") if b.strip()]
    for block in blocks:
        nb.cells.append(nbformat.v4.new_code_cell(block))

    path = os.path.join(results_dir, "analysis.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return path
