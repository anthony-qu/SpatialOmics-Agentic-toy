import json
import anthropic

CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192


def _build_header(data_context: dict) -> str:
    paths = [f["path"] for f in data_context["files"]]
    return f"""import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_FILES = {paths}
RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)
"""


def _check_syntax(code: str):
    """Return error message if code has a syntax error, else None."""
    try:
        compile(code, "<generated>", "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


def _extract_code(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    if "```" in text:
        parts = text.split("```")
        # parts[1] is the code block content
        code = parts[1]
        if code.startswith("python"):
            code = code[6:]
        return code.strip()
    return text.strip()


def generate(plan: dict, data_context: dict) -> str:
    """
    LLM call: plan + data_context → executable Python code string.
    Includes a self-critique round before returning.
    """
    system_prompt = open("prompts/system_coder.txt").read()
    system_prompt = system_prompt.replace(
        "{{DATA_CONTEXT}}", json.dumps(data_context, indent=2)
    )

    user_message = f"""Analysis plan:
{json.dumps(plan, indent=2)}

Write the complete Python code to execute this plan."""

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    code = _extract_code(response.content[0].text)

    # Syntax check — catches truncated output before wasting execution retries
    syntax_error = _check_syntax(code)
    if syntax_error:
        fix_response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {
                    "role": "user",
                    "content": (
                        f"The code has a syntax error ({syntax_error}), likely because it was truncated. "
                        "Rewrite the complete code. Avoid long hardcoded lists — use short representative "
                        "examples or programmatic approaches instead."
                    ),
                },
            ],
        )
        code = _extract_code(fix_response.content[0].text)

    # Self-critique round
    critique_response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=512,
        system="You are a code reviewer for bioinformatics scripts. Be concise.",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Data context:\n{json.dumps(data_context, indent=2)}\n\n"
                    f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
                    f"Generated code:\n```python\n{code}\n```\n\n"
                    "Does this code correctly implement the plan given the data context? "
                    "Check column names, file paths (it should use DATA_FILES variable), and figure filenames. "
                    "Reply with OK if correct, or list specific issues if not."
                ),
            }
        ],
    )

    critique = critique_response.content[0].text.strip()

    if critique.upper() != "OK" and not critique.upper().startswith("OK"):
        # One fix pass
        fix_response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {
                    "role": "user",
                    "content": f"The code has these issues:\n{critique}\n\nFix them and return the complete corrected code.",
                },
            ],
        )
        code = _extract_code(fix_response.content[0].text)

    return _build_header(data_context) + "\n" + code


def fix(code: str, error: str, data_context: dict) -> str:
    """
    LLM call: broken code + traceback → fixed code.
    Called by agent.py when executor reports failure.
    """
    system_prompt = open("prompts/system_coder.txt").read()
    system_prompt = system_prompt.replace(
        "{{DATA_CONTEXT}}", json.dumps(data_context, indent=2)
    )

    is_syntax = "SyntaxError" in error
    extra_instruction = (
        " Avoid long hardcoded lists — use short representative examples or programmatic approaches instead."
        if is_syntax else ""
    )

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"This code failed with the following error:\n\n"
                    f"ERROR:\n{error}\n\n"
                    f"CODE:\n```python\n{code}\n```\n\n"
                    f"Fix the code and return the complete corrected version.{extra_instruction}"
                ),
            }
        ],
    )

    fixed = _extract_code(response.content[0].text)

    # Catch syntax errors in the fix itself before returning
    syntax_error = _check_syntax(fixed)
    if syntax_error:
        retry = CLIENT.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"This code has a syntax error ({syntax_error}), likely due to truncation. "
                        "Rewrite it concisely. Avoid long hardcoded lists.\n\n"
                        f"```python\n{fixed}\n```"
                    ),
                }
            ],
        )
        fixed = _extract_code(retry.content[0].text)

    # Re-prepend header in case LLM dropped it
    if "DATA_FILES" not in fixed:
        fixed = _build_header(data_context) + "\n" + fixed
    return fixed
