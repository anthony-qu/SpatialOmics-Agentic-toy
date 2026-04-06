import argparse
import glob
import os
import scanpy as sc

import planner
import coder
import executor
import critic
import notebook_writer


def inspect_data(data_path: str) -> dict:
    """Read metadata from all .h5ad files in data_path. No LLM involved."""
    pattern = os.path.join(os.path.abspath(data_path), "*.h5ad")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .h5ad files found in {data_path}")

    file_contexts = []
    for path in files:
        adata = sc.read_h5ad(path)
        file_contexts.append({
            "path": path,
            "label": os.path.splitext(os.path.basename(path))[0],
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obs_columns": list(adata.obs.columns),
            "obsm_keys": list(adata.obsm.keys()),
            "uns_keys": list(adata.uns.keys()),
            "has_spatial": "spatial" in adata.uns or "spatial" in adata.obsm,
        })

    return {"files": file_contexts}


def run(user_prompt: str, data_path: str) -> None:
    print("=== Step 1: Inspecting data ===")
    data_context = inspect_data(data_path)
    for f in data_context["files"]:
        print(f"  {f['label']}: {f['n_obs']} obs × {f['n_vars']} vars | spatial={f['has_spatial']}")

    print("\n=== Step 2: Planning ===")
    plan = planner.run(user_prompt, data_context)
    print(f"  Goal: {plan['goal']}")
    for step in plan["steps"]:
        fig = f" → {step['figure_filename']}" if step.get("produces_figure") else ""
        print(f"  [{step['step_id']}] {step['description']}{fig}")

    print("\n=== Step 3: Generating code ===")
    code = coder.generate(plan, data_context)
    print("  Code generated and self-critiqued.")

    print("\n=== Step 4: Executing ===")
    MAX_RETRIES = 3
    result = executor.run_code(code)

    retries = 0
    while not result["success"] and retries < MAX_RETRIES:
        retries += 1
        print(f"  Execution failed (attempt {retries}/{MAX_RETRIES}). Fixing...")
        print(f"  Error: {result['error'][:300]}")
        code = coder.fix(code, result["error"], data_context)
        result = executor.run_code(code)

    if not result["success"]:
        print(f"  Execution failed after {MAX_RETRIES} retries.")
        print(f"  Last error: {result['error']}")
        return

    if result["stdout"]:
        print(result["stdout"])

    print("\n=== Step 4b: Saving notebook ===")
    try:
        nb_path = notebook_writer.save_as_notebook(code, plan)
        print(f"  Notebook saved: {nb_path}")
    except Exception as e:
        print(f"  Warning: could not save notebook: {e}")

    print("\n=== Step 5: Critiquing figures ===")
    critique = critic.run(plan["figure_filenames"], user_prompt)
    for v in critique["verdicts"]:
        print(f"  [{v['verdict']}] {v['filename']}: {v['reason']}")

    print(f"\n=== Done. {critique['summary']} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy agentic scRNA-seq analysis system.")
    parser.add_argument("user_prompt", help="What you want to analyze.")
    parser.add_argument("data_path", help="Path to directory containing .h5ad files.")
    args = parser.parse_args()
    run(args.user_prompt, args.data_path)
