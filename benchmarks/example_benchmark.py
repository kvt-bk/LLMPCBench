from benchmarks.base_benchmark import BaseBenchmark

class ExampleBenchmark(BaseBenchmark):
    """
    A simple example benchmark with a few factual questions.
    """
    def __init__(self):
        super().__init__("Simple QA")
        self.questions = [
            {
                "id": 1,
                "prompt": "What is the capital of France?",
                "expected_answer_keywords": ["paris"] # Case-insensitive check
            },
            {
                "id": 2,
                "prompt": "Who wrote 'Romeo and Juliet'?",
                "expected_answer_keywords": ["shakespeare", "william"]
            },
            {
                "id": 3,
                "prompt": "What is 2 + 2?",
                "expected_answer_keywords": ["4", "four"]
            }
        ]

    def get_questions(self):
        """Returns the predefined list of questions."""
        return self.questions

    def evaluate(self, model_response: str, question_data: dict) -> float:
        """
        Evaluates if the model's response contains the expected keywords.
        Returns 1.0 if all keywords are found (case-insensitive), 0.0 otherwise.
        """
        if not model_response:
            return 0.0
        
        response_lower = model_response.lower()
        expected_keywords = question_data.get("expected_answer_keywords", [])
        
        if not expected_keywords: # Should not happen if questions are defined correctly
            return 0.0 
            
        all_keywords_found = all(keyword.lower() in response_lower for keyword in expected_keywords)
        
        return 1.0 if all_keywords_found else 0.0