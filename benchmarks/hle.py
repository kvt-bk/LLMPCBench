# ollama_eval_project/benchmarks/hle_adapter.py
from benchmarks.base_benchmark import BaseBenchmark

class HLEAdapter(BaseBenchmark):
    def __init__(self):
        super().__init__("Holistic Language Evaluation (HLE)")
        
        print(f"NOTICE: {self.name} adapter is a stub. Define your HLE tasks and evaluation metrics.")
        self.questions = [
            {
                "id": "hle_reasoning_1",
                "prompt": "If a train leaves City A at 10:00 AM traveling at 60 mph, and City B is 180 miles away, what time will it arrive in City B, assuming no stops?",
                "type": "reasoning", # You might have different types of questions
                "expected_keywords": ["1:00 pm", "13:00"]
            },
            {
                "id": "hle_creative_1",
                "prompt": "Write a very short story (2-3 sentences) about a curious cat exploring a new room.",
                "type": "creative_writing",
                # Evaluation for creative tasks is harder, might be qualitative or use LLM-as-judge. For now, simple length check.
                "min_sentences": 2
            }
        ]


    def get_questions(self):
        # TODO: Load or define HLE tasks/questions.
        print(f"WARNING: {self.name}.get_questions() uses example questions.")
        return self.questions

    def evaluate(self, model_response: str, question_data: dict) -> (float | None):
        # TODO: Implement evaluation logic based on the type of HLE task.
        print(f"WARNING: {self.name}.evaluate() uses example logic for question ID {question_data.get('id')}.")
        q_type = question_data.get("type")
        if q_type == "reasoning":
            response_lower = model_response.lower()
            return 1.0 if any(kw in response_lower for kw in question_data.get("expected_keywords", [])) else 0.0
        elif q_type == "creative_writing":
            # Simple check: number of sentences (approx by periods, question marks, exclamations)
            sentences = model_response.count('.') + model_response.count('!') + model_response.count('?')
            return 1.0 if sentences >= question_data.get("min_sentences", 2) else 0.0
        return 0.0 # Default score if type not handled