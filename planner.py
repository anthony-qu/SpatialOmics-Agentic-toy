import json
import anthropic

CLIENT = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def run(user_prompt: str, data_context: dict) -> dict:
    """
    LLM call: user_prompt + data_context → JSON analysis plan.
    Returns dict with keys: goal, steps, figure_filenames.
    """
    system_prompt = open("prompts/system_planner.txt").read()
    system_prompt = system_prompt.replace("{{DATA_CONTEXT}}", json.dumps(data_context, indent=2))

    user_message = f"User request: {user_prompt}"

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        # One retry: ask the LLM to fix its own output
        fix_response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a JSON repair assistant. Return only valid JSON, no explanation.",
            messages=[
                {"role": "user", "content": f"This JSON is malformed. Fix it:\n\n{raw}"}
            ],
        )
        plan = json.loads(fix_response.content[0].text.strip())

    return plan
