import queue
import jupyter_client


def run_code(code: str, timeout: int = 120) -> dict:
    """
    Run a code string in a fresh Jupyter kernel.
    Returns {"success": bool, "stdout": str, "stderr": str, "error": str | None}
    """
    km = jupyter_client.KernelManager(kernel_name="python3")
    km.start_kernel()
    kc = km.blocking_client()
    kc.start_channels()
    kc.wait_for_ready(timeout=30)

    stdout_parts = []
    stderr_parts = []
    error = None

    try:
        msg_id = kc.execute(code)

        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                error = "Execution timed out."
                break

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                if content["name"] == "stdout":
                    stdout_parts.append(content["text"])
                elif content["name"] == "stderr":
                    stderr_parts.append(content["text"])

            elif msg_type == "error":
                error = "\n".join(content["traceback"])

            elif msg_type == "status" and content["execution_state"] == "idle":
                # Check if this idle is for our execute request
                try:
                    reply = kc.get_shell_msg(timeout=5)
                    if reply["parent_header"].get("msg_id") == msg_id:
                        break
                except queue.Empty:
                    break

    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    return {
        "success": error is None,
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "error": error,
    }
