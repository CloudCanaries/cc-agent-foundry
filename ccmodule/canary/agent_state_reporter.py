"""
AgentStateReporter - Mixin for worker agents to report OODA loop states.

This class should be mixed into all worker agent classes to enable
real-time state reporting to the backend.
"""

import logging
import os
import requests
import sys
import datetime
from typing import Dict, Optional
from .utils import LoggerMixin


AGENT_STATE_UPDATE = "canaryableservice/agent-state/"


class AgentStateReporter(LoggerMixin, object):
    """
    Mixin class for worker agents to report their OODA loop states to the backend.

    Usage:
        class MyAgent(AgentStateReporter):
            def __init__(self):
                super().__init__()
                self.agent_type = "MY_AGENT_TYPE"

            def run(self):
                self.transition_to(self.STATE_OBSERVE, "Starting observation...")
                # ... do work ...
                self.transition_to(self.STATE_ORIENT, "Analyzing findings...")
                # ... do work ...
    """

    STATE_OBSERVE = "OBSERVE"
    STATE_ORIENT = "ORIENT"
    STATE_DECIDE = "DECIDE"
    STATE_ACT = "ACT"

    def __init__(self):
        """Initialize state reporter configuration."""
        self.backend_url = os.getenv("CANARY_API_SERVER")
        self.state_endpoint = f"{self.backend_url}/{AGENT_STATE_UPDATE}"

        self.api_key = os.getenv("API_KEY")

        self.canary_id: Optional[str] = os.getenv("CANARY_ID")
        self.agent_type: Optional[str] = None

        self.current_state: str = self.STATE_OBSERVE
        self.last_report_time: Optional[datetime.datetime] = None

        self.enable_state_reporting = os.getenv(
            "ENABLE_AGENT_STATE_REPORTING", "true"
        ).lower() in ("true", "1", "yes")
        self.report_timeout = int(os.getenv("AGENT_STATE_REPORT_TIMEOUT", "5"))

        # Setup logging with agent-specific metadata
        self._logger = logging.getLogger(self.__class__.__name__)
        self._set_logger_meta()

    def _set_logger_meta(self):
        """Configure logger with agent-specific metadata."""
        agent_type = getattr(self, "CANARY_TYPE", "UNKNOWN_AGENT")
        canary_id = getattr(self, "CANARY_ID", "NO_CANARY_ID")
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)d - "
            + f"[{agent_type}|{canary_id[:8]}] "
            + "%(message)s"
        )
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def transition_to(
        self,
        new_state: str,
        action_description: str = "",
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Transition to a new OODA loop state and report to backend.

        Args:
            new_state: One of STATE_OBSERVE, STATE_ORIENT, STATE_DECIDE, STATE_ACT
            action_description: Human-readable description of current action
            metadata: Additional context (progress, details, etc.)

        Returns:
            True if state was successfully reported, False otherwise
        """
        old_state = self.current_state
        self.current_state = new_state

        self._logger.info(
            f"[{self.agent_type}] State transition: {old_state} -> {new_state} "
            f"({action_description})"
        )

        report_metadata = metadata or {}
        report_metadata["current_action"] = action_description
        report_metadata["previous_state"] = old_state
        report_metadata["transition_time"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        return self.report_state(metadata=report_metadata)

    def report_state(self, metadata: Optional[Dict] = None) -> bool:
        """
        Report current state to backend.

        Args:
            metadata: Optional metadata about current state

        Returns:
            True if successfully reported, False otherwise
        """
        if not self.enable_state_reporting:
            self._logger.debug("[AgentStateReporter] State reporting is disabled")
            return True

        if not self.canary_id:
            self._logger.warning(
                "[AgentStateReporter] canary_id not set, cannot report state"
            )
            return False

        if not self.api_key:
            self._logger.warning(
                "[AgentStateReporter] ORGANIZATION_API_KEY not set, cannot report state"
            )
            return False

        payload = {
            "agent_id": self.canary_id,
            "agent_type": self.agent_type or "UNKNOWN",
            "ooda_state": self.current_state,
            "metadata": metadata or {},
        }

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.state_endpoint,
                json=payload,
                headers=headers,
                timeout=self.report_timeout,
            )

            if response.status_code == 200:
                self.last_report_time = datetime.datetime.now(datetime.timezone.utc)
                return True
            else:
                self._logger.error(
                    f"[AgentStateReporter] Failed to report state: "
                    f"status={response.status_code}, body={response.text}"
                )
                return False

        except requests.exceptions.Timeout:
            self._logger.warning(
                f"[AgentStateReporter] Timeout reporting state to {self.state_endpoint}"
            )
            return False
        except requests.exceptions.RequestException as e:
            self._logger.error(f"[AgentStateReporter] Error reporting state: {e}")
            return False

    def observe(self, action: str = "", metadata: Optional[Dict] = None) -> bool:
        """Transition to OBSERVE state."""
        return self.transition_to(self.STATE_OBSERVE, action, metadata)

    def orient(self, action: str = "", metadata: Optional[Dict] = None) -> bool:
        """Transition to ORIENT state."""
        return self.transition_to(self.STATE_ORIENT, action, metadata)

    def decide(self, action: str = "", metadata: Optional[Dict] = None) -> bool:
        """Transition to DECIDE state."""
        return self.transition_to(self.STATE_DECIDE, action, metadata)

    def act(self, action: str = "", metadata: Optional[Dict] = None) -> bool:
        """Transition to ACT state."""
        return self.transition_to(self.STATE_ACT, action, metadata)
