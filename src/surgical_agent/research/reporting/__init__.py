"""Event reporting and downstream frame-level GSR public API."""

VERSION = "gsr_report_v1"

from surgical_agent.research.reporting.contracts import (
    EventReportGenerator,
    ReportRecord,
)
from surgical_agent.research.reporting.manager import EventReportManager
from surgical_agent.research.reporting.template import TemplateReportGenerator
from surgical_agent.research.reporting.writer import EventReportWriter

__all__ = [
    "EventReportGenerator",
    "EventReportManager",
    "EventReportWriter",
    "ReportRecord",
    "TemplateReportGenerator",
]
