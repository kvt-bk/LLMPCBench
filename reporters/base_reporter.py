from abc import ABC, abstractmethod

class BaseReporter(ABC):
    """Abstract base class for all result reporters."""
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def report(self, results_data: list[dict]):
        """
        Process and output the evaluation results.

        Args:
            results_data (list[dict]): A list of result dictionaries from the evaluator.
        """
        pass