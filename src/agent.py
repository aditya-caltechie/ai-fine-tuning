from __future__ import annotations

import logging


class Agent:
    """
    Minimal base Agent class (logging + simple styling).

    Kept tiny on purpose — just enough for SpecialistAgent logging.
    """

    # ANSI colors (used by the notebook logs)
    RED = "\033[31m"
    RESET = "\033[0m"
    BG_BLACK = "\033[40m"

    name: str = "Agent"
    color: str = RED

    def log(self, message: str) -> None:
        prefix = f"[{self.name}]"
        formatted = f"{self.BG_BLACK}{self.color}{prefix} {message}{self.RESET}"
        logging.getLogger().info(formatted)

