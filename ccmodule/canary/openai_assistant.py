from __future__ import annotations
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union, TypedDict
from openai import OpenAI
from croniter import croniter
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
        if OpenAI is None:
            raise ValueError(
                "The 'openai' package is not installed. "
                "Install the extra: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.assistant_id = assistant_id or os.getenv("OPENAI_ASSISTANT_ID")
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries

        """Setup Logging"""
        self._logger = self.get_logger(self.__class__.__name__)
        self._set_logger_meta()

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.assistant_id:
            raise ValueError("OPENAI_ASSISTANT_ID is required")

        self.client = client or OpenAI(api_key=self.api_key)

    def _set_logger_meta(self):
        asst = getattr(self, "assistant_id", None) or "UNKNOWN_ASSISTANT"
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)d - "
            + asst
            + " %(message)s"
        )
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
            self._logger.error(
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

    def _get_ai_assistant_enabled(self) -> bool:
        return os.getenv("AI_ASSISTANT_ENABLED", "true").lower() == "true"

    def _get_autonomous_mode_enabled(self) -> bool:
        """
        Gate for autonomous actions. Defaults to disabled unless explicitly turned on.
        """
        return os.getenv("AUTONOMOUS_MODE_ENABLED", "false").lower() == "true"

    def _get_ai_assistant_cron(self) -> str:
        return os.getenv("AI_ASSISTANT_CRON", "0 */12 * * *")

    def _should_run_assistant_now(self) -> bool:
        """
        Return True when the current UTC time is sufficiently close to a cron slot.
        """
        if not self._get_ai_assistant_enabled():
            return False

        cron_expr = self._get_ai_assistant_cron().strip()
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        try:
            itr = croniter(cron_expr, now)
            prev_slot = itr.get_prev(datetime)
            next_slot = itr.get_next(datetime)

            if not prev_slot or not next_slot:
                return False

            interval = (next_slot - prev_slot).total_seconds()
            base_tolerance = interval * 0.1 if interval > 0 else 30
            tolerance_seconds = max(15, min(90, base_tolerance))

            delta_prev = abs((now - prev_slot).total_seconds())
            delta_next = abs((next_slot - now).total_seconds())
            delta_min = min(delta_prev, delta_next)

            return delta_min <= tolerance_seconds
        except Exception:
            return False

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
        allow_when_autonomous_disabled: bool = True,
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
        allow_when_autonomous_disabled : bool
            If False and AUTONOMOUS_MODE_ENABLED is off, returns `default` without calling the assistant.

        Returns
        -------
        str | dict | list
            The assistant's text reply, or parsed JSON if `parse_json` is True.
        """
        if (
            not self._get_autonomous_mode_enabled()
            and not allow_when_autonomous_disabled
        ):
            self._logger.info(
                "Autonomous mode disabled; skipping OpenAI assistant execution."
            )
            return default

        if not hasattr(self, "_oaiclient"):
            self._init_openai_assistant()

        if short_recommendations:
            base_instructions = (
                "You are a Senior Site Reliability Engineer. Your task is to "
                "analyze the provided data and offer concise, actionable "
                "recommendations for resolving any identified issues. "
                "Keep your recommendations brief and to the point."
            )
            merged = (
                f"{instructions.strip()}\n\n{base_instructions}"
                if instructions
                else base_instructions
            )
            return self._oaiclient.run_text(
                payload, instructions=merged, metadata=metadata
            )
        if parse_json:
            return self._oaiclient.run_json(
                payload, instructions=instructions, metadata=metadata, default=default
            )

        return self._oaiclient.run_text(
            payload, instructions=instructions, metadata=metadata
        )

    def build_markdown_assistant_instructions(
        self, *, agent_role: str, input_desc: str, focus_areas: list[str]
    ) -> str:
        """
        Build standardized assistant instructions for agents that return
        Markdown bullet-point recommendations.

        Args:
            agent_role (str): The agent role (e.g., "AWS IAM/Least-Privilege expert").
            input_desc (str): Short description of the INPUT_JSON context.
            focus_areas (list[str]): List of key focus areas/actions as strings.

        Returns:
            str: Formatted assistant instructions string.
        """
        focus_str = "\n".join([f"- {area}" for area in focus_areas])

        template = (
            "You are an on-call {ROLE}. Analyze INPUT_JSON: {INPUT_DESCRIPTION}.\n"
            "\n"
            "Return the output in **Markdown** only — **exactly 5 lines**, each a bullet starting with `- `.\n"
            "Do **not** include JSON (even inside code fences), numbered lists, code fences, headings, or any extra text.\n"
            "Each bullet: start with a **bolded issue summary**, then a colon, then 1-2 concrete fixes.\n"
            "Allowed Markdown: bullet lists (- …), **bold**, `inline code`, and standard links.\n"
            "\n"
            "Focus on concrete actions such as:\n"
            "{FOCUS_AREAS}\n"
            "\n"
            "Output format example (structure only; generate your own content):\n"
            "- **Issue summary**: specific, actionable fix 1 or fix 2.\n"
            "- **Another problem**: suggested concrete remediation.\n"
            "- **Configuration gap**: step to harden or remediate.\n"
            "- **Risk detected**: action to mitigate.\n"
            "- **Performance/coverage issue**: fix or scaling adjustment.\n"
        )

        return template.format(
            ROLE=agent_role,
            INPUT_DESCRIPTION=input_desc,
            FOCUS_AREAS=focus_str,
        )
