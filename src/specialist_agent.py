from __future__ import annotations

import modal

from agent import Agent


class SpecialistAgent(Agent):
    """
    Calls the deployed Modal class `Pricer` from `pricer_service2.py`.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self) -> None:
        self.log("Specialist Agent is initializing - connecting to modal")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        self.log("Specialist Agent is calling remote fine-tuned model")
        # Stream remote container logs (including print() calls) to the local terminal.
        with modal.enable_output():
            result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return float(result)
