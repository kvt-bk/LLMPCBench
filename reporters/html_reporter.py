import os
import logging
from datetime import datetime
from reporters.base_reporter import BaseReporter

logger = logging.getLogger(__name__)

# (HTML templates can be defined here as in the previous version)
HTML_BASE_TEMPLATE = """...""" # Same as before
RUN_TEMPLATE = """..."""       # Same as before

class HTMLReporter(BaseReporter):
    """Saves evaluation results to a cumulative HTML file."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.output_filename = self.config.get('output_file', 'evaluation_results.html')

    def report(self, results_data: list[dict]):
        # The logic from the previous save_results_to_html function goes here
        # This implementation remains largely the same, just wrapped in a class.
        if not results_data:
            logger.warning("No results were generated, skipping HTML report generation.")
            return

        # ... (Insert the full logic of the previous save_results_to_html here)
        # For brevity, I'm omitting the full copy-paste, but it's the same function.
        # It should generate the HTML block for the current run and append it.
        
        logger.info(f"HTML report updated: {self.output_filename}")