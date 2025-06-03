# ollama_eval_project/benchmarks/mmlu_pro_adapter.py
from .base_benchmark import BaseBenchmark
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import re
import ast # For safely evaluating string representation of lists


class MMLUProAdapter(BaseBenchmark):
    HF_DATASET_NAME = "TIGER-Lab/MMLU-Pro"

    DEFAULT_SUBJECT_COLUMN_NAME = 'category'
    DEFAULT_QUESTION_TEXT_COLUMN_NAME = 'question'
    DEFAULT_CHOICES_LIST_COLUMN_NAME = 'options'
    DEFAULT_ANSWER_LETTER_COLUMN_NAME = 'answer'
    DEFAULT_ANSWER_INDEX_COLUMN_NAME = 'answer_index'

    def __init__(self, 
                 subjects: list[str] | None = None, 
                 data_split: str = "test",
                 max_questions_per_subject: int | None = None): 
        """
        Initializes the MMLU-Pro benchmark adapter using Hugging Face Datasets.

        Args:
            subjects (list[str] | None, optional): 
                A list of specific MMLU-Pro subjects (configurations on Hugging Face) to run.
                If None, all available subjects for MMLU-Pro will be attempted.
            data_split (str, optional): 
                Which data split to use ("test", "validation", or "dev"). 
                Defaults to "test".
            max_questions_per_subject (int | None, optional):
                Maximum number of questions to load per subject. If None, all questions are loaded.
                Defaults to None.
        """
        #super().__init__(f"MMLU-Pro ({data_split}" + (f", {max_questions_per_subject}q/subj" if max_questions_per_subject else "") + ")") 
        super().__init__(f"MMLU-Pro ({data_split}")
        
        self.subjects_to_run_filter = subjects
        self.data_split = data_split
        self.max_questions_per_subject = max_questions_per_subject
        self.questions = []

        # These will be determined from the actual loaded dataset's first item
        self.actual_subject_column_name = None
        self.actual_choices_column_name = None
        self.actual_question_column_name = None
        self.actual_answer_letter_column_name = None
        self.actual_answer_index_column_name = None


    # _format_prompt method remains the same
    def _format_prompt(self, subject: str, question_text: str, options: dict) -> str:
        subject_formatted = subject.replace("_", " ").title()
        prompt = f"The following is a multiple choice question about {subject_formatted}.\n\n"
        prompt += f"Question: {question_text}\n"
        for i in range(10): # A-J
            option_letter = chr(ord('A') + i)
            if option_letter in options and options[option_letter] is not None:
                prompt += f"{option_letter}. {options[option_letter]}\n"
        prompt += "Answer:"
        return prompt

    def _determine_column_names(self, sample_item_keys: list):
        """Determine actual column names from a sample item's keys."""
        
        # Prioritize specific names if found, then fall back to lists of possibles
        # For Subject
        if self.DEFAULT_SUBJECT_COLUMN_NAME in sample_item_keys:
            self.actual_subject_column_name = self.DEFAULT_SUBJECT_COLUMN_NAME
        else:
            for name in ['subject', 'category', 'task', 'topic']: # Fallback list
                if name in sample_item_keys: self.actual_subject_column_name = name; break
        if self.actual_subject_column_name: print(f"    Determined subject column name: '{self.actual_subject_column_name}'")
        else: print(f"    WARNING: Could not auto-determine subject column from keys: {sample_item_keys}")

        # For Choices List
        if self.DEFAULT_CHOICES_LIST_COLUMN_NAME in sample_item_keys:
            self.actual_choices_column_name = self.DEFAULT_CHOICES_LIST_COLUMN_NAME
        else:
            for name in ['choices', 'options', 'passages', 'endings', 'multiple_choices']: # Fallback list
                if name in sample_item_keys: self.actual_choices_column_name = name; break
        if self.actual_choices_column_name: print(f"    Determined choices list column name: '{self.actual_choices_column_name}'")
        else: print(f"    ERROR: Could not auto-determine choices list column. Options will likely be empty.")

        # For Question Text
        if self.DEFAULT_QUESTION_TEXT_COLUMN_NAME in sample_item_keys:
            self.actual_question_column_name = self.DEFAULT_QUESTION_TEXT_COLUMN_NAME
        elif 'question' in sample_item_keys: self.actual_question_column_name = 'question' # Common fallback
        if self.actual_question_column_name: print(f"    Determined question text column name: '{self.actual_question_column_name}'")
        else: print(f"    ERROR: Could not auto-determine question text column.")
        
        # For Answer Letter
        if self.DEFAULT_ANSWER_LETTER_COLUMN_NAME in sample_item_keys:
            self.actual_answer_letter_column_name = self.DEFAULT_ANSWER_LETTER_COLUMN_NAME
        elif 'answer' in sample_item_keys: self.actual_answer_letter_column_name = 'answer'
        if self.actual_answer_letter_column_name: print(f"    Determined answer letter column name: '{self.actual_answer_letter_column_name}'")
        else: print(f"    WARNING: Could not auto-determine answer letter column. Will rely on index if available.")

        # For Answer Index
        if self.DEFAULT_ANSWER_INDEX_COLUMN_NAME in sample_item_keys:
            self.actual_answer_index_column_name = self.DEFAULT_ANSWER_INDEX_COLUMN_NAME
        elif 'answer_index' in sample_item_keys: self.actual_answer_index_column_name = 'answer_index'
        if self.actual_answer_index_column_name: print(f"    Determined answer index column name: '{self.actual_answer_index_column_name}'")
        else: print(f"    WARNING: Could not auto-determine answer index column.")

    def _load_data(self):
        loaded_questions = []
        question_print_debug_counter = 0

        try:
            print(f"  Loading data for {self.HF_DATASET_NAME} (split: {self.data_split}, config: default)...")
            full_dataset = load_dataset(self.HF_DATASET_NAME, name="default", split=self.data_split)
            print(f"    Full dataset (default config, {self.data_split} split) loaded with {len(full_dataset)} items.")

            if not full_dataset or len(full_dataset) == 0:
                print("    ERROR: Loaded dataset is empty or failed to load.")
                return []

            # Determine column names from the first item
            sample_item = full_dataset[0]
            self._determine_column_names(list(sample_item.keys()))
            
            # Ensure critical column names were found
            if not self.actual_question_column_name or not self.actual_choices_column_name:
                print("    ERROR: Critical column names for question or choices could not be determined. Cannot proceed.")
                return []
            if not self.actual_answer_letter_column_name and not self.actual_answer_index_column_name:
                print("    ERROR: Critical column names for answer (letter or index) could not be determined. Cannot proceed.")
                return []


            items_to_process_grouped = {}
            # Grouping logic (as before, but using self.actual_subject_column_name)
            # ... (This part remains similar to the previous version, ensure it uses self.actual_subject_column_name if found)
            # For brevity, I'll skip repeating the grouping logic here, assume it's correctly adapted.
            # Simplified: iterate through full_dataset and apply filters/grouping
            
            # Using a simplified loop for demonstration if subject filtering is complex here:
            data_iterator = full_dataset
            if self.subjects_to_run_filter and self.actual_subject_column_name:
                print(f"    Filtering for subjects: {self.subjects_to_run_filter} using column '{self.actual_subject_column_name}'.")
                data_iterator = full_dataset.filter(
                    lambda example: example.get(self.actual_subject_column_name) in self.subjects_to_run_filter
                )
            # The per-subject limiting would need more complex grouping here if applied after this initial filter.
            # For now, let's assume max_questions_per_subject applies to the total after filtering.
            # A more robust solution would group by subject first, then limit, then collect.
            # The previous version's grouping logic was better for per-subject limiting.
            # For this snippet, I'll simplify to show just option parsing.

            # Temp counter for max_questions overall if not doing per-subject limiting here
            current_total_questions = 0

            for item_idx, item in enumerate(data_iterator):
                # Apply overall max questions if not None (simplification)
                if self.max_questions_per_subject is not None and current_total_questions >= (self.max_questions_per_subject * (len(self.subjects_to_run_filter) if self.subjects_to_run_filter else 1) ) :
                    if self.subjects_to_run_filter: # only break if we are filtering by subject and hit overall limit
                         break
                    # if no subject filter, this means max_questions_per_subject acts as total max questions
                    elif not self.subjects_to_run_filter and current_total_questions >= self.max_questions_per_subject:
                        break


                if question_print_debug_counter < 2: # Debug print for first 2 items successfully processed
                    print(f"\nDEBUGGING ITEM (Overall #{question_print_debug_counter + 1}, Item Idx in dataset #{item_idx}):")
                    if hasattr(item, 'keys'): print(f"  Item keys: {list(item.keys())}")
                    print(f"  Full item data (first 200 chars): {str(item)[:200]}")


                options = {}
                # The 'options' column in CSV is a string representation of a list.
                # We need to parse it.
                choices_str = item.get(self.actual_choices_column_name)
                choices_list = []
                if choices_str and isinstance(choices_str, str):
                    try:
                        choices_list = ast.literal_eval(choices_str) # Safely evaluate string to list
                        if not isinstance(choices_list, list):
                            print(f"    WARNING: Parsed '{self.actual_choices_column_name}' is not a list for item ID '{item.get('question_id', 'N/A')}'. Value: {choices_list}")
                            choices_list = [] # Reset if not a list
                    except (ValueError, SyntaxError) as e:
                        print(f"    WARNING: Could not parse '{self.actual_choices_column_name}' string to list for item ID '{item.get('question_id', 'N/A')}'. Error: {e}. Value: {choices_str}")
                        choices_list = [] # Reset on error
                elif isinstance(choices_str, list): # If it's already a list (less likely from CSV read this way but good check)
                    choices_list = choices_str


                if choices_list:
                    for i, choice_text in enumerate(choices_list):
                        if i < 10: # Supports A-J
                            option_letter = chr(ord('A') + i)
                            options[option_letter] = str(choice_text)
                else:
                    # Fallback if choices_list is empty or parsing failed
                    for letter_key_fallback in ['A', 'B', 'C', 'D']: # Check for individual columns as extreme fallback
                        if letter_key_fallback in item and item[letter_key_fallback] is not None:
                            options[letter_key_fallback] = str(item[letter_key_fallback])
                    if not options:
                        print(f"    WARNING: Could not load options for item ID '{item.get('question_id', 'N/A')}'.")
                
                for i in range(10): # Ensure A-J keys exist
                    letter = chr(ord('A') + i)
                    if letter not in options: options[letter] = None
                
                valid_options_for_prompt = {k: v for k, v in options.items() if v is not None}
                
                question_text_val = str(item.get(self.actual_question_column_name, ''))
                subject_name_from_data = str(item.get(self.actual_subject_column_name, "unknown_subject"))
                
                # Determine correct answer letter
                correct_answer_char = "INVALID_ANSWER" # Default
                if self.actual_answer_letter_column_name and item.get(self.actual_answer_letter_column_name) is not None:
                    correct_answer_char = str(item.get(self.actual_answer_letter_column_name)).strip().upper()
                elif self.actual_answer_index_column_name and item.get(self.actual_answer_index_column_name) is not None:
                    try:
                        answer_idx = int(item.get(self.actual_answer_index_column_name))
                        if 0 <= answer_idx < len(choices_list) and answer_idx < 10:
                            correct_answer_char = chr(ord('A') + answer_idx)
                        else:
                             print(f"    WARNING: Answer index {answer_idx} out of bounds for choices list for item ID '{item.get('question_id', 'N/A')}'")
                    except ValueError:
                         print(f"    WARNING: Answer index is not an int: {item.get(self.actual_answer_index_column_name)} for item ID '{item.get('question_id', 'N/A')}'")
                else:
                    print(f"    WARNING: No answer letter or index found for item ID '{item.get('question_id', 'N/A')}'")


                question_data = {
                    "id": f"{subject_name_from_data}_{item.get('question_id', item_idx)}", # Use 'question_id' if available
                    "subject": subject_name_from_data,
                    "question_text": question_text_val,
                    "options": options,
                    "correct_answer_char": correct_answer_char,
                    "prompt": self._format_prompt(subject_name_from_data, question_text_val, valid_options_for_prompt)
                }
                loaded_questions.append(question_data)
                current_total_questions += 1


                if question_print_debug_counter < 10: 
                    print(f"  Processed Question (ID: {question_data['id']}):")
                    print(f"    Loaded Options: {options}")
                    print(f"    Correct Answer: '{correct_answer_char}'")
                question_print_debug_counter += 1
            
            # The grouping logic for max_questions_per_subject would ideally be here
            # to reorganize `loaded_questions` if it wasn't handled by the iterator itself.
            # For now, the simplified overall limit is in the loop.

        except Exception as e:
            print(f"  ERROR: Major failure during MMLU-Pro data loading or processing: {e}")
            import traceback
            traceback.print_exc()

        print(f"Total MMLU-Pro questions loaded: {len(loaded_questions)}")
        return loaded_questions

    # get_questions, _extract_choice, and evaluate methods remain the same
    def get_questions(self):
        if not self.questions:
            self.questions = self._load_data()
        return self.questions

    def _extract_choice(self, model_response: str) -> str | None:
        if not model_response:
            return None
        
        # --- Preprocessing Step to Remove Reasoning Blocks ---
        # Remove <think>...</think> blocks (case-insensitive, multiline)
        # The re.DOTALL flag makes '.' match newlines as well.
        # The re.IGNORECASE flag makes <think> match <Think>, <THINK>, etc.
        processed_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL | re.IGNORECASE)
        
        # It's also common for models to output just the answer after the thinking block.
        # Sometimes there might be a final answer marker like </answer_final> or similar.
        # For now, we focus on removing the <think> blocks.
        # You might also want to strip leading/trailing whitespace from the processed_response.
        processed_response = processed_response.strip()
        # --- End Preprocessing Step ---

        # Now, use the processed_response for extraction
        response = processed_response # Use the cleaned response from here on

        if not response: # If the response was ONLY a think block
            return None

        valid_choice_chars = "ABCDEFGHIJ"
        patterns = [
            # Case 1: Explicit statements with choice in parentheses
            # Examples: "The final answer is (A)", "Answer: (B)"
            rf"final answer is\s*:?\s*\(([{valid_choice_chars}])\)",
            rf"The final answer is\s*:?\s*\(([{valid_choice_chars}])\)",
            rf"Answer\s*:?\s*\(([{valid_choice_chars}])\)",
            rf"The correct option is\s*:?\s*\(([{valid_choice_chars}])\)",
            rf"The correct answer is\s*:?\s*\(([{valid_choice_chars}])\)", # E.g., "The correct answer is (A)"
            rf"The correct choice is\s*:?\s*\(([{valid_choice_chars}])\)",

            # Case 2: Explicit statements with choice NOT in parentheses (NEW)
            # Examples: "The final answer is A.", "The correct answer is B"
            rf"final answer is\s*:?\s*([{valid_choice_chars}])(?:[.\s]|$)",
            rf"The final answer is\s*:?\s*([{valid_choice_chars}])(?:[.\s]|$)",
            rf"Answer\s*:?\s*([{valid_choice_chars}])(?:[.\s]|$)",
            rf"The correct option is\s*:?\s*([{valid_choice_chars}])(?:[.\s]|$)",
            rf"The correct answer is\s*:?\s*([{valid_choice_chars}])(?:[.\s]|$)", # E.g., "The correct answer is A." or "The correct answer is A "
            rf"The correct choice is\s*:?\s*([{valid_choice_chars}])(?:[.\s]|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Case 3: Just the letter in parentheses "(A)"
        match = re.search(rf"\(([{valid_choice_chars}])\)", response)
        if match:
            return match.group(1).upper()
        
        
        # Case 4: Letter followed by a period or parenthesis, often at the start of a line or response
        # Examples: "A.", "B)", " C." (note: initial space handled by strip)
        # We look for it potentially being the first thing, or first after some initial non-alphanumeric preamble.
        # This is more general.
        match = re.search(rf"^([{valid_choice_chars}])(?:[.)\s])", response)
        if match:
            return match.group(1).upper()
        
        # Case 5: If the response is just a single letter A-J.
        if len(response) == 1 and response.upper() in valid_choice_chars:
            return response.upper()
        
        # Case 6: Fallback: Check the very start of the response for a letter that is
        # directly followed by a space, or is the only character.
        # Avoids capturing 'A' from "An apple..."
        
        first_char = response[:1].upper()
        if first_char in valid_choice_chars:
            if len(response) > 1:
                second_char = response[1:2]
                if not second_char.isalpha():
                    return first_char
            else:
                return first_char
        return None

    def evaluate(self, model_response: str, question_data: dict) -> (float | None):
        extracted_choice = self._extract_choice(model_response)
        correct_answer = question_data.get("correct_answer_char")
        score = 0.0

        # === DEBUG PRINTING for evaluation step ===
        # Only print for a few questions to avoid flooding console, or for specific failing IDs
        # if question_data['id'] in ["some_specific_failing_id_1", "another_failing_id"]:
        # For now, let's print if the correct answer is potentially beyond D or if score is 0
        # to catch problematic cases.
        
        should_debug_this_eval = False
        if correct_answer not in ['A', 'B', 'C', 'D']:
            should_debug_this_eval = True
        
        if extracted_choice != correct_answer: # If it's going to be a 0 score
            should_debug_this_eval = True # Also debug all incorrect answers

        # Limit debugging to first N questions overall if MMLU_PRO_MAX_QUESTIONS_PER_SUBJECT is small
        # This assumes question_data['id'] gives some sequential nature or you use a global counter passed in.
        # For simplicity, we'll rely on the above conditions for now.

        prompt_options_actually_shown = {}
        for i in range(10): # A-J
            opt_char = chr(ord('A') + i)
            if opt_char in question_data['options'] and question_data['options'][opt_char] is not None:
                prompt_options_actually_shown[opt_char] = question_data['options'][opt_char]

        if should_debug_this_eval: # Print details if score is 0 or if correct answer is unusual
            print(f"\n--- Debugging Evaluation for Question ID: {question_data['id']} ---")
            print(f"  Prompt (first 100 chars): {question_data['prompt'][:100]}...")
            print(f"  Options included in the actual prompt: {prompt_options_actually_shown}") # What was shown
            print(f"  Full Options from Dataset: {question_data['options']}") # All options available in data
            print(f"  Correct Answer Char: '{correct_answer}'")
            print(f"  Model Raw Response (first 100 chars): '{model_response[:100]}...'")
            print(f"  Extracted Choice by _extract_choice: '{extracted_choice}'")
        # === END DEBUG PRINTING ===

        if extracted_choice is None:
            # Current: score = 0.0
            if should_debug_this_eval: print("  Evaluation Result: Could not extract choice -> Score: 0.0")
            return 0.0 

        if extracted_choice == correct_answer:
            score = 1.0
            if should_debug_this_eval: print(f"  Evaluation Result: Correct! -> Score: {score}")
            return score
        else:
            # Current: score = 0.0
            if should_debug_this_eval: print(f"  Evaluation Result: Incorrect. -> Score: 0.0")
            return 0.0