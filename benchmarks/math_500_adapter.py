# ollama_eval_project/benchmarks/math_500_adapter.py
from .base_benchmark import BaseBenchmark
import re

class Math500Adapter(BaseBenchmark):
    def __init__(self):
        super().__init__("MATH-500 (Subset)")
        # TODO: Full MATH-500 integration requires dataset and potentially specific parsing for answers.
        # The MATH dataset problems are complex and often require step-by-step reasoning.
        # Evaluation often involves checking the final numerical answer.
        print(f"NOTICE: {self.name} adapter is a stub. Full implementation requires the MATH dataset and robust answer extraction.")
        self.questions = [
            {
                "id": "math_q1",
                "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "latex_solution": "In April, Natalia sold 48 clips. In May, she sold half as many clips as in April, so she sold $48/2 = 24$ clips. Altogether, in April and May, she sold $48 + 24 = 72$ clips. The final answer is $\boxed{72}$.",
                "numerical_answer": "72"
            },
            {
                "id": "math_q2",
                "prompt": "What is the value of $3 + 5 \\times 2$?",
                "latex_solution": "$3 + 5 \\times 2 = 3 + 10 = 13$. The final answer is $\boxed{13}$.",
                "numerical_answer": "13"
            }
            # Add more representative questions if developing further
        ]

    def get_questions(self):
        print(f"WARNING: {self.name}.get_questions() uses example questions.")
        return self.questions

    def _extract_answer(self, model_response: str) -> str | None:
        """
        A simple attempt to extract a final numerical answer.
        Robust answer extraction from LLM math responses is a research problem.
        This might look for "The final answer is X" or the last number in the response.
        """
        # Try to find numbers preceded by "answer is", "result is", etc.
        patterns = [
            r"answer is\s*([+-]?\d+\.?\d*)",
            r"result is\s*([+-]?\d+\.?\d*)",
            r"\boxed{([+-]?\d+\.?\d*)}", # LaTeX box
            r"([+-]?\d+\.?\d*)\s*$", # Last number in the string
        ]
        for pattern in patterns:
            match = re.search(pattern, model_response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: find any number in the last line
        last_line = model_response.strip().split('\n')[-1]
        numbers = re.findall(r"([+-]?\d+\.?\d*)", last_line)
        if numbers:
            return numbers[-1]
            
        return None


    def evaluate(self, model_response: str, question_data: dict) -> (float | None):
        print(f"WARNING: {self.name}.evaluate() uses example logic for question ID {question_data.get('id')}.")
        if not model_response:
            return 0.0
            
        extracted_answer = self._extract_answer(model_response)
        expected_answer = question_data.get("numerical_answer")

        if extracted_answer is not None and expected_answer is not None:
            try:
                # Compare as floats if they might have decimals, or as integers
                if "." in extracted_answer or "." in expected_answer:
                    return 1.0 if abs(float(extracted_answer) - float(expected_answer)) < 1e-3 else 0.0
                else:
                    return 1.0 if int(extracted_answer) == int(expected_answer) else 0.0
            except ValueError:
                # Could not convert extracted answer to number
                return 0.0
        return 0.0