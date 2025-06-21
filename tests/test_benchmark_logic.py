import pytest
from benchmarks.mmlu_pro import MMLUProAdapter

# We need an instance to test the private method
# In a real scenario, you might initialize it with minimal/mock data
adapter = MMLUProAdapter()

@pytest.mark.parametrize("response, expected", [
    ('Some thinking... <think>I will choose B.</think> The final answer is {"Answer": "B"}.', 'B'),
    ('{"Answer": "C"}', 'C'),
    ('The correct option is A.', 'A'),
    ('Based on my analysis, the answer is: (D)', 'D'),
    ('I am confident the choice is E', None), # No clear keyword
    ('<think>The answer is F</think>', None), # Stripped out
    ('Answer: G', 'G'),
    ('json: {"answer": "h"}', 'H'), # Case-insensitive key and value
])
def test_extract_choice(response, expected):
    """Tests the _extract_choice logic with various response formats."""
    assert adapter._extract_choice(response) == expected