from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing import List, Optional, Dict, Any, Union
import requests

from .utils import LoggerMixin

WORKFLOW_INGEST_URL = "canaryableservice/ingest-workflow/"


class ActionOption(str, Enum):
    """Canonical action options."""

    NONE = ""
    APPROVE = "approve"
    ROLLBACK = "rollback"
    CONFIRM = "confirm"


@dataclass
class WorkflowStep:
    """Single workflow step payload."""

    main_title: str
    step_number: int
    subtitle1: str = ""
    subtitle2: str = ""
    action_option: str = ActionOption.NONE.value
    action_required: bool = False
    interval_seconds: int = 0
    extra_information: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return {
            "subtitle1": self.subtitle1,
            "subtitle2": self.subtitle2,
            "main_title": self.main_title,
            "step_number": self.step_number,
            "action_option": self.action_option,
            "action_required": self.action_required,
            "interval_seconds": self.interval_seconds,
            "extra_information": self.extra_information,
        }


class WorkflowConstructMixin(LoggerMixin, object):
    """
    Mixin to construct and POST workflow steps JSON to the platform.
    Usage:
        class MyWorkflow(WorkflowConstructMixin):
            def __init__(self):
                super().__init__(
                    workflow_name="My Workflow",
                    trigger="Triggered by X event",
                    description="Short description of the workflow"
                )
                self.add_step(main_title="Step 1", ...)
                ...
                response = self.post_workflow(endpoint="https://api.cloudcanaries.ai")
    Args:
        workflow_name: Name/label for the workflow.
        trigger: Human-readable trigger/condition text.
        description: Short description of the workflow.
    """

    def __init__(
        self,
        *,
        workflow_name: str,
        trigger: str,
        description: str,
    ) -> None:
        """
        Args:
            workflow_name: Name/label for the workflow.
            trigger: Human-readable trigger/condition text.
            description: Short description of the workflow.
        """
        self._workflow_name = workflow_name
        self._trigger = trigger
        self._description = description

        self._steps: List[WorkflowStep] = []
        self._next_step_number: int = 1
        self._logger = self.get_logger(self.__class__.__name__)

    def add_step(
        self,
        *,
        main_title: str,
        subtitle1: str = "",
        subtitle2: str = "",
        action_option: Union[str, ActionOption] = ActionOption.NONE,
        action_required: bool = False,
        interval_seconds: int = 0,
        extra_information: str = "",
        step_number: Optional[int] = None,
    ) -> "WorkflowConstructMixin":
        if not main_title or not main_title.strip():
            raise ValueError("main_title is required and cannot be empty.")

        if isinstance(action_option, ActionOption):
            action_option = action_option.value
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be >= 0.")

        number = step_number if step_number is not None else self._next_step_number

        step = WorkflowStep(
            main_title=main_title.strip(),
            subtitle1=subtitle1.strip(),
            subtitle2=subtitle2.strip(),
            action_option=str(action_option).strip(),
            action_required=bool(action_required),
            interval_seconds=int(interval_seconds),
            extra_information=str(extra_information).strip(),
            step_number=int(number),
        )
        self._steps.append(step)
        self._next_step_number = max(self._next_step_number, number + 1)
        return self

    def build_workflow(self) -> Dict[str, Any]:
        """
        Construct the final JSON-ready workflow object.

        Returns:
            dict with keys: name, steps, trigger, description
        """
        if not self._steps:
            self._logger.warning("Building workflow with zero steps.")
        ordered = sorted(self._steps, key=lambda s: s.step_number)
        return {
            "name": self._workflow_name,
            "steps": [s.to_payload() for s in ordered],
            "trigger": self._trigger,
            "description": self._description,
        }

    def post_workflow(
        self,
        *,
        endpoint: str,
        timeout: int = 15,
        headers: Optional[Dict[str, str]] = None,
        raise_for_status: bool = True,
    ) -> Dict[str, Any]:
        """
        POST the built workflow JSON to the platform.

        Args:
            endpoint: Base endpoint (e.g., 'https://api.cloudcanaries.ai').
            timeout: Requests timeout (seconds).
            headers: Authorization headers to merge with defaults.
            raise_for_status: If True, raise for non-2xx responses.

        Returns:
            Response JSON (parsed dict). If body is empty/non-JSON,
            returns {"status_code": <int>}.
        """
        url = f"{endpoint}/{WORKFLOW_INGEST_URL}"

        payload = self.build_workflow()
        final_headers = {"Content-Type": "application/json", **(headers or {})}

        resp = requests.post(url, headers=final_headers, json=payload, timeout=timeout)
        if raise_for_status:
            resp.raise_for_status()

        try:
            return resp.json()
        except ValueError:
            return {"status_code": resp.status_code}

    def reset_workflow(self) -> None:
        """Clear all steps and reset numbering to 1."""
        self._steps.clear()
        self._next_step_number = 1

    def set_metadata(
        self,
        *,
        name: Optional[str] = None,
        trigger: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Update workflow-level metadata."""
        if name is not None:
            self._workflow_name = name
        if trigger is not None:
            self._trigger = trigger
        if description is not None:
            self._description = description

    def add_step_initialize_snapshot(self, source_desc: str, interval: int = 5):
        return self.add_step(
            main_title="Initialize & Snapshot",
            subtitle1=f"Collecting current state from {source_desc}",
            subtitle2="Establish baseline for diff & risk scoring",
            interval_seconds=interval,
            action_required=False,
        )

    def add_step_analyze_plan(self, method_desc: str, interval: int = 8):
        return self.add_step(
            main_title="Analyze & Plan",
            subtitle1=f"Diagnosing issues via {method_desc}",
            subtitle2="Derive prioritized actions with risk tags",
            interval_seconds=interval,
            action_required=False,
        )

    def add_step_ai_assist(
        self, assistant_desc: str = "OpenAI Assistant", interval: int = 10
    ):
        return self.add_step(
            main_title="AI-Assist Recommendations",
            subtitle1=f"Querying {assistant_desc} for validated actions (JSON only)",
            subtitle2="Guardrails: deterministic schema, least-privilege actions",
            interval_seconds=interval,
            action_required=False,
        )

    def add_step_human_gate(
        self, label: str = "Approve Remediation Plan", interval: int = 0
    ):
        return self.add_step(
            main_title="Human Approval Gate",
            subtitle1=label,
            subtitle2="Proceed only after explicit approval",
            action_required=True,
            action_option=ActionOption.APPROVE.value,
            interval_seconds=interval,
        )

    def add_step_execute(self, label: str = "Apply Remediations", interval: int = 20):
        return self.add_step(
            main_title=label,
            subtitle1="Applying actions with idempotent safe-guards",
            subtitle2="Rollback on failure where supported",
            interval_seconds=interval,
            action_required=False,
        )

    def add_step_verify(
        self, label: str = "Post-Remediation Verification", interval: int = 12
    ):
        return self.add_step(
            main_title=label,
            subtitle1="Re-scan resources and compare before/after",
            subtitle2="Generate metrics and compliance deltas",
            interval_seconds=interval,
            action_required=False,
        )

    def add_step_notify(self, channels: str = "Slack & Email", interval: int = 3):
        return self.add_step(
            main_title="Notify Stakeholders",
            subtitle1=f"Send summary to {channels}",
            subtitle2="Include links to logs, PRs, dashboards",
            interval_seconds=interval,
            action_required=False,
        )

    def add_step_close(self, interval: int = 2):
        return self.add_step(
            main_title="Close & Archive",
            subtitle1="Persist artifacts, metrics, and evidence",
            subtitle2="Mark workflow as complete",
            interval_seconds=interval,
            action_required=False,
        )

    def add_preset_ai_remediation_pipeline(
        self,
        *,
        snapshot_source: str,
        analysis_method: str,
        needs_human_gate: bool = True,
        execute_label: str = "Apply Remediations",
        include_notify: bool = True,
    ) -> "WorkflowConstructMixin":
        """
        Compose a consistent end-to-end pipeline.
        """
        self.add_step_initialize_snapshot(snapshot_source)
        self.add_step_analyze_plan(analysis_method)
        self.add_step_ai_assist()
        if needs_human_gate:
            self.add_step_human_gate()
        self.add_step_execute(execute_label)
        self.add_step_verify()
        if include_notify:
            self.add_step_notify()
        self.add_step_close()
        return self
