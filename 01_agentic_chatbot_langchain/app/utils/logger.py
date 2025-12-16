
import logging
import json
import os
from langchain.callbacks.base import BaseCallbackHandler

###################################################
# Logging

def get_logger(name="agent_usage"):
    """
    Create and return a logger for agent usage events. Ensures no duplicate handlers.
    """
    log_file = os.path.join(os.getcwd(), "agent_usage.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    logger.info("Logger started")
    return logger


class UsageLoggerHandler(BaseCallbackHandler):
    def __init__(self, logger=None, json_path="usage_log.json", **kwargs):
        self.logger = logger
        self.json_path = json_path
        self.current_step = kwargs.get("step", None)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _append_to_json(self, data: dict):
        """Append record vào JSON file."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        existing_data.append(data)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Khi tool bắt đầu chạy"""
        self.current_step = serialized.get("name", "unknown_step")

    def on_llm_end(self, response, **kwargs):
        """Khi LLM trả về kết quả"""
        text = response.generations[0][0].text if response.generations else ""
        usage = response.llm_output.get("token_usage", {})

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Cộng dồn token
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        log_data = {
            "event": "llm_end",
            "run_id": str(kwargs.get("run_id", "")),
            "parent_run_id": str(kwargs.get("parent_run_id", "")),
            "step": self.current_step,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "output_text": text,
            "cumulative_input_tokens": self.total_input_tokens,
            "cumulative_output_tokens": self.total_output_tokens
        }

        if self.logger:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))

        self._append_to_json(log_data)

    def on_agent_action(self, action, **kwargs):
        """Log mỗi step agent thực hiện"""
        log_data = {
            "event": "agent_action",
            "run_id": str(kwargs.get("run_id", "")),
            "parent_run_id": str(kwargs.get("parent_run_id", "")),
            "tool": action.tool,
            "tool_input": str(action.tool_input),
            "log": action.log
        }

        if self.logger:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))

        self._append_to_json(log_data)

    def on_tool_end(self, output, **kwargs):
        """Log kết quả tool trả về"""
        log_data = {
            "event": "tool_end",
            "run_id": str(kwargs.get("run_id", "")),
            "parent_run_id": str(kwargs.get("parent_run_id", "")),
            "output": str(output)
        }

        if self.logger:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))

        self._append_to_json(log_data)

    def on_chain_end(self, outputs, **kwargs):
        """Log khi một chain kết thúc"""
        log_data = {
            "event": "chain_end",
            "run_id": str(kwargs.get("run_id", "")),
            "parent_run_id": str(kwargs.get("parent_run_id", "")),
            "outputs": outputs
        }

        if self.logger:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))

        self._append_to_json(log_data)

logger = get_logger()
