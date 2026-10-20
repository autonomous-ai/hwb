"""Exception hierarchy. Anything raised on purpose by hwb derives from HwbError."""

from __future__ import annotations


class HwbError(Exception):
    """Expected, user-facing failure. The CLI prints the message without a traceback."""


class ToolMissingError(HwbError):
    def __init__(self, tool: str) -> None:
        super().__init__(f"required tool not found on PATH: {tool}")
        self.tool = tool


class UnsupportedPlatformError(HwbError):
    pass


class UserAbortError(HwbError):
    pass
