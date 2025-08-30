from __future__ import annotations
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union, TypedDict
from openai import OpenAI
from .utils import LoggerMixin


# Set up logging
def configure_global_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)s +++- %(message)s",
    )


configure_global_logger()
logger = logging.getLogger("GlobalLogger")


def _strip_md_fences(text: str) -> str:
    """
    Remove leading/trailing Markdown fences from LLM outputs.

    Args:
        text: The raw assistant output, wrapped in ```json ... ```.

    Returns:
        Cleaned text without backtick fences.
    """
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"^```[\w-]*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned


class OpenAIAssistantClient(LoggerMixin, object):
    """
    Small helper around the OpenAI Assistants (Threads) API.

    Parameters
    ----------
    api_key : Optional[str]
        OpenAI API key. Falls back to env `OPENAI_API_KEY` if omitted.
    assistant_id : Optional[str]
        Target Assistant ID. Falls back to env `OPENAI_ASSISTANT_ID` if omitted.
    client : Optional[OpenAI]
        Pre-initialized OpenAI client. If omitted, one is created.
    request_timeout_s : int
        Per-request timeout seconds (best-effort).
    max_retries : int
        How many times to re-list messages after a run finishes (race-avoidance).
    logger : Optional[Any]
        Logger with .info/.warning/.error. If provided, logs helpful details.

    Raises
    ------
    ValueError
        If OpenAI is unavailable or required env vars are missing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        assistant_id: Optional[str] = None,
        client: Optional[OpenAI] = None,
        request_timeout_s: int = 120,
        max_retries: int = 2,
    ):
        """Setup Logging"""
        self._logger = self.get_logger(self.__class__.__name__)
        self._set_logger_meta()
        if OpenAI is None:
            raise ValueError(
                "The 'openai' package is not installed. "
                "Install the extra: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.assistant_id = assistant_id or os.getenv("OPENAI_ASSISTANT_ID")
        self.client = client or OpenAI(api_key=self.api_key)
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.logger = logger

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.assistant_id:
            raise ValueError("OPENAI_ASSISTANT_ID is required")

    def _set_logger_meta(self):

        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)d - "
            + self.assistant_id
            + " %(message)s"
        )
        # Remove existing handlers to prevent duplicate logs
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    # ---------- public helpers ----------

    def run_text(
        self,
        payload: Union[str, Dict[str, Any], Sequence[Any]],
        *,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send payload to the Assistant and return the latest text message.

        Parameters
        ----------
        payload : str | dict | list
            Snapshot/context to analyze, or a free-form string.
        instructions : Optional[str]
            Extra guidance (e.g., “Return ONLY valid JSON”).
        metadata : Optional[Dict[str, Any]]
            Metadata stored on the thread.

        Returns
        -------
        str
            The assistant's first available text block (or empty string).
        """
        msg = self._to_message_content(payload, instructions=instructions)
        return self._run_and_read_text(msg, metadata=metadata)

    def run_json(
        self,
        payload: Union[str, Dict[str, Any], Sequence[Any]],
        *,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        default: Union[Dict[str, Any], List[Any], str] = "[]",
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Send payload and parse the reply as JSON (tolerant to fenced outputs).

        Parameters
        ----------
        payload : str | dict | list
            Snapshot/context to analyze.
        instructions : Optional[str]
            Guidance like “Return ONLY a JSON array of actions”.
        metadata : Optional[Dict[str, Any]]
            Metadata stored on the thread.
        default : dict | list | str
            Value returned when JSON parsing fails.

        Returns
        -------
        dict | list | str
            Parsed JSON on success, otherwise `default`.
        """
        text = self.run_text(payload, instructions=instructions, metadata=metadata)
        cleaned = _strip_md_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if self.logger:
                self.logger.error(
                    "Failed to parse assistant JSON; returning default. Raw: %s",
                    cleaned[:500],
                )
            return default

    # ---------- internal ----------

    def _run_and_read_text(
        self,
        message_text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        thread = self.client.beta.threads.create(metadata=metadata or {})
        self.client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=message_text
        )
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=self.assistant_id
        )

        # Get messages; allow a couple of retries (avoids race conditions)
        for _ in range(self.max_retries + 1):
            msgs = list(
                self.client.beta.threads.messages.list(
                    thread_id=thread.id, run_id=run.id
                )
            )
            txt = self._first_text(msgs)
            if txt:
                return txt
            time.sleep(0.6)

        msgs = list(self.client.beta.threads.messages.list(thread_id=thread.id))
        return self._first_text(msgs) or ""

    @staticmethod
    def _first_text(messages) -> Optional[str]:
        """Return the first text value from a messages list, if any."""
        for m in messages:  # API usually returns newest-first
            for part in getattr(m, "content", []) or []:
                if getattr(part, "type", None) == "text":
                    return getattr(part.text, "value", None) or ""
        return None

    @staticmethod
    def _to_message_content(
        payload: Union[str, Dict[str, Any], Sequence[Any]],
        *,
        instructions: Optional[str] = None,
    ) -> str:
        """
        Prepare a single text message for the Assistant.

        Returns
        -------
        str
            Instructions (if present) + a pretty-printed JSON block or raw text.
        """
        if isinstance(payload, str):
            body = payload
        else:
            body = json.dumps(payload, default=str, indent=2)
        return (
            f"{instructions.strip()}\n\n<INPUT_JSON>\n{body}\n</INPUT_JSON>"
            if instructions
            else body
        )


class OpenAIAssistantMixin:
    """
    Mixin wiring an `OpenAIAssistantClient` from env vars.

    Expected on the parent:
    - self._logger (from your LoggerMixin)
    """

    _oaiclient: OpenAIAssistantClient

    def _init_openai_assistant(
        self,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        asst_id_env: str = "OPENAI_ASSISTANT_ID",
    ) -> None:
        """
        Initialize the assistant client from env variables.

        Raises
        ------
        ValueError
            If env variables are missing or `openai` is not installed.
        """
        api_key = os.getenv(api_key_env)
        assistant_id = os.getenv(asst_id_env)
        self._oaiclient = OpenAIAssistantClient(
            api_key=api_key,
            assistant_id=assistant_id,
            client=None,
        )

    def send_to_assistant(
        self,
        payload: Union[str, Dict[str, Any], Sequence[Any]],
        *,
        instructions: Optional[str] = None,
        parse_json: bool = True,
        default: Union[Dict[str, Any], List[Any], str] = "[]",
        metadata: Optional[Dict[str, Any]] = None,
        short_recommendations: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Dispatch helper used by agents.

        Parameters
        ----------
        payload : str | dict | list
            The snapshot / data to analyze.
        instructions : Optional[str]
            Optional guidance, e.g. “Return ONLY a JSON array of actions”.
        parse_json : bool
            Whether to parse JSON from the reply.
        default : dict | list | str
            Fallback value if JSON parsing fails.
        metadata : Optional[Dict[str, Any]]
            Optional metadata for the thread.
        short_recommendations : bool
            If True, prepends instructions to keep recommendations short.

        Returns
        -------
        str | dict | list
            The assistant's text reply, or parsed JSON if `parse_json` is True.
        """
        if not hasattr(self, "_oaiclient"):
            self._init_openai_assistant()

        if short_recommendations:
            return self._oaiclient.run_text(
                payload,
                instructions=(
                    "Summarize INPUT_JSON into a short, user-friendly "
                    "recommendations section. Keep it concise, no JSON, "
                    "no preamble, just the text."
                ),
                metadata=metadata,
            )
        if parse_json:
            return self._oaiclient.run_json(
                payload, instructions=instructions, metadata=metadata, default=default
            )

        return self._oaiclient.run_text(
            payload, instructions=instructions, metadata=metadata
        )
