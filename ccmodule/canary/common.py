import datetime
import logging
import os
import signal
import sys
import time
import uuid
from datetime import timedelta
from enum import Enum

import requests
from croniter import croniter

from .utils import EnvVarValidatorMixin
from .utils import LoggerMixin

"""
Important ENV_VARS:

API_KEY - authenticates with api server[REQUIRED]
CANARY_ID - sets the canary id to uniquely associate data[REQUIRED]
CANARY_API_SERVER - location of reporting server[REQUIRED]
"""


# Set up logging
def configure_global_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)s +++- %(message)s",
    )


configure_global_logger()
logger = logging.getLogger("GlobalLogger")


class CanaryRunStatus(Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    NEW = "NEW"


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def roundDownTime(dt=None, dateDelta=timedelta(minutes=1)):
    roundTo = dateDelta.total_seconds()
    if dt == None:
        dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def getNextCronRunTime(schedule):
    return croniter(schedule, datetime.datetime.now()).get_next(datetime.datetime)


def sleepTillTopOfNextMinute():
    t = datetime.datetime.utcnow()
    sleeptime = 60 - (t.second + t.microsecond / 1000000.0)
    time.sleep(sleeptime)


try:
    sys.path.append("/CC")
except Exception as e:
    logger.error("Error adding path %s", e)

try:
    sys.path.append(
        os.path.join(
            "./",
        )
    )
except Exception as e:
    logger.error("Error adding path %s", e)


"""
    Common methods for both canaries here and how to interact with the Cloud Canaries System
"""


class CanaryBasePrototype(EnvVarValidatorMixin, LoggerMixin, object):
    """The base class for a Canary"""

    first_run = True

    def __init__(self):
        """Setup Logging"""
        self._logger = self.get_logger(self.__class__.__name__)
        self._set_logger_meta()

        """Validate required environment variables"""
        env_vars = self.validate_env_vars(
            ["CANARY_API_SERVER", "API_KEY", "CANARY_ID"], strict=False
        )

        self._error_get = False
        self._error_parse = False
        self._uuid = str(uuid.uuid4())
        self.status = CanaryRunStatus.NEW

    def _set_logger_meta(self):
        canary_id = self._canary_id
        dag_id = self._canary_dag_id
        run_id = self._canary_run_id
        run_try_number = self._canary_run_try_number

        canary_identifier_fmt = (
            "[canary_id=%s][dag_id=%s][run_id=%s][run_try_number=%s]"
            % (canary_id, dag_id, run_id, run_try_number)
        )

        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)d - "
            + canary_identifier_fmt
            + " %(message)s"
        )
        # Remove existing handlers to prevent duplicate logs
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    @property
    def _api_key(self):
        return os.getenv("API_KEY")

    @property
    def _canary_run_id(self):
        return os.getenv("RUN_ID") or "CANARY_RUN_ID_NOT_FOUND"

    @property
    def _canary_run_try_number(self):
        return os.getenv("TRY_NUMBER") or "CANARY_TRY_NUMBER_NOT_FOUND"

    @property
    def _canary_dag_id(self):
        return os.getenv("DAG_ID") or "CANARY_DAG_ID_NOT_FOUND"

    @property
    def _canaryServer(self):
        """configured via env var"""
        return os.getenv("CANARY_API_SERVER")

    @property
    def _canary_id(self):
        return os.getenv("CANARY_ID")

    @property
    def _is_agent_canary(self):
        is_agent_canary = os.getenv("IS_AGENT_CANARY", "").lower()
        return is_agent_canary == "true"

    @property
    def _canary_schedule(self):
        canary_schedule = os.getenv("CANARY_SCHEDULE")
        if not canary_schedule:
            raise ValueError("Environment variable CANARY_SCHEDULE must be set")
        return canary_schedule

    # , '0 0 1 1 *'

    def get_response(self):
        """overwritten by actual canary"""
        raise NotImplementedError

    def parse_response(self, response):
        """overwritten by actual canary"""
        raise NotImplementedError

    def _report(self, error_get=False, error_parse=False, value=None, latency_ms=None):
        """reports results back to canary server"""
        url = os.path.join(self._canaryServer, "canarydata/")

        self._logger.info(
            " _report "
            "[error_get=%s][error_parse=%r][value=%r][url=%s][latency_ms=%s]",
            error_get,
            error_parse,
            value,
            url,
            latency_ms,
        )

        try:
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            _json = {
                "error_get": error_get,
                "error_parse": error_parse,
                "value": value,
                "response_latency": latency_ms,
                "canary_id": self._canary_id,
                "dag_id": self._canary_dag_id,
                "run_id": self._canary_run_id,
                "run_try_number": self._canary_run_try_number,
                "measured": datetime.datetime.utcnow().isoformat(),
                "downtime_generating_error": error_get,
            }
            headers = {"Authorization": "Api-Key %s" % self._api_key}
            response = requests.post(url, json=_json, headers=headers)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                self._logger.error(
                    "exception in reporting [error=%s][status_code=%s]",
                    e,
                    response.status_code,
                )
                raise e
            self._logger.info("reported [status_code=%s]", response.status_code)

        except Exception as e:
            self._logger.error(
                "exception in _report "
                "[error_get=%s][error_parse=%r][value=%r][latency_ms=%s][error=%s]",
                error_get,
                error_parse,
                value,
                latency_ms,
                e,
            )

    def run(self):
        if self._is_agent_canary:
            self._logger.info(
                "running canary in agent mode [canary_id=%s][canary_schedule=%s]",
                self._canary_id,
                self._canary_schedule,
            )
            killer = GracefulKiller()
            nextRunTime = getNextCronRunTime(self._canary_schedule)
            while not killer.kill_now:
                roundedDownTime = roundDownTime()
                if roundedDownTime == nextRunTime:
                    if self.first_run:
                        self.report_status(CanaryRunStatus.RUNNING)
                        self.first_run = False
                    self._logger.info(
                        "running canary [roundedDownTime=%s][nextRunTime=%s]",
                        roundedDownTime,
                        nextRunTime,
                    )
                    self._run()
                    nextRunTime = getNextCronRunTime(self._canary_schedule)
                elif roundedDownTime > nextRunTime:
                    nextRunTime = getNextCronRunTime(self._canary_schedule)
                sleepTillTopOfNextMinute()
            self.report_status(CanaryRunStatus.STOPPED)
        else:
            self._logger.info(
                "running canary in non-agent mode [canary_id=%s]",
                self._canary_id,
            )
            self._run()

    def _run(self):
        """gets a response,
        parses it,
        reports parsed results back to server
        """
        value = None
        latency_ms = None
        try:
            self._logger.info("trying get_response")
            start = time.perf_counter()
            response = self.get_response()
            self._logger.info("raw response %s", response)
            self._logger.info("response dir %s", dir(response))

            stop = time.perf_counter()
            latency_ms = round((stop - start) * 1000)
            self._logger.info("get_response [error=False][latency_ms=%s]", latency_ms)
        except Exception as e:
            self._logger.exception("get_response [error=True] %s", e)
            self._error_get = True
        if not self._error_get:
            try:
                self._logger.info("trying parse_response")
                value = self.parse_response(response)
                self._logger.info("parse_response [error=False][value=%r]", value)
            except Exception as e:
                self._error_parse = True
                self._logger.error("parse_response [error=True] %s", e)

        self._report(self._error_get, self._error_parse, value, latency_ms)

        self.latency_ms = latency_ms
        self.reported_value = value

    def _report_canary_agent_status(self):
        """reports canary agent status to canary server"""
        url = os.path.join(
            self._canaryServer, f"canary/{self._canary_id}/canary_agent_status/"
        )

        self._logger.info(
            f"_report_canary_agent_status [url={url}][status={self.status}]"
        )

        try:
            _json = {"id": self._canary_id, "status": self.status.value}
            headers = {"Authorization": "Api-Key %s" % self._api_key}

            response = requests.patch(url, json=_json, headers=headers)

            self._logger.info(
                "status changed status [status_code=%s]", response.status_code
            )

        except Exception as e:
            self._logger.error(f"exception in _report_canary_agent_status [error={e}]")

    def report_status(self, status):
        if status == CanaryRunStatus.STOPPED:
            self.status = CanaryRunStatus.STOPPED
        elif status == CanaryRunStatus.RUNNING:
            self.status = CanaryRunStatus.RUNNING
        elif status == CanaryRunStatus.ERROR:
            self.status = CanaryRunStatus.ERROR
        elif status == CanaryRunStatus.NEW:
            self.status = CanaryRunStatus.NEW
        else:
            self.status = CanaryRunStatus.NEW

        self._report_canary_agent_status()
