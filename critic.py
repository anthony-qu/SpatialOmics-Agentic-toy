import base64
import os
import anthropic

CLIENT = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"
RESULTS_DIR = "results/"


def run(figure_filenames: list[str], user_prompt: str) -> dict:
    """
    VLM call: evaluate each saved figure.
    Returns {"verdicts": [{"filename": ..., "verdict": "PASS"|"FAIL", "reason": ...}], "summary": str}
    """
    system_prompt = open("prompts/system_critic.txt").read()
    verdicts = []

    for filename in figure_filenames:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            verdicts.append({
                "filename": filename,
                "verdict": "FAIL",
                "reason": f"File not found: {path}",
            })
            continue

        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=256,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"User goal: {user_prompt}\n\n"
                                f"Figure filename: {filename}\n\n"
                                "Does this figure look scientifically reasonable? "
                                "Check for: blank axes, single cluster, all-zero data, missing labels, obvious errors. "
                                "Reply with PASS or FAIL followed by one sentence of reasoning."
                            ),
                        },
                    ],
                }
            ],
        )

        reply = response.content[0].text.strip()
        verdict = "PASS" if reply.upper().startswith("PASS") else "FAIL"
        verdicts.append({"filename": filename, "verdict": verdict, "reason": reply})

    passed = sum(1 for v in verdicts if v["verdict"] == "PASS")
    summary = f"{passed}/{len(verdicts)} figures passed critique."

    return {"verdicts": verdicts, "summary": summary}
