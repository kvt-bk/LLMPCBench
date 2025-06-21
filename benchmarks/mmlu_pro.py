from benchmarks.base_benchmark import BaseBenchmark
from datasets import load_dataset
import re
import ast # For safely evaluating string representation of lists from CSVs (if ever used as fallback)
import json
import logging
import math
logger = logging.getLogger(__name__)

class MMLUPro(BaseBenchmark):
    HF_DATASET_NAME = "TIGER-Lab/MMLU-Pro"

    # Define known column names from TIGER-Lab/MMLU-Pro dataset on Hugging Face
    # Based on sample-test-data.csv and typical structure.
    COL_QUESTION_ID = 'question_id' # Or 'idx' if that's more common in HF dataset
    COL_QUESTION_TEXT = 'question'
    COL_CHOICES_LIST = 'options' # This is a list of strings
    COL_ANSWER_LETTER = 'answer' # The letter, e.g., 'A', 'I'
    COL_ANSWER_INDEX = 'answer_index' # The 0-based index
    COL_CATEGORY = 'category' # Subject identifier

    def __init__(self,
                 subjects: list[str] | None = None,
                 data_split: str = "test",
                 percentage_per_subject: float | None = None):
        super().__init__(f"MMLU-Pro ({data_split})")


        self.subjects_to_run_filter = subjects
        self.data_split = data_split
        self.percentage_per_subject = percentage_per_subject
        self.questions = [] # Cache for loaded questions

        notice_msg = f"NOTICE: {self.name} adapter initialized for HF dataset ({self.HF_DATASET_NAME}, default config)."
        if self.subjects_to_run_filter:
            notice_msg += f" Will filter for specified subjects: {self.subjects_to_run_filter}."
        logging.info(notice_msg)

    def _format_prompt(self, subject: str, question_text: str, options: dict) -> str:
        """
        Formats a zero-shot prompt for an MMLU-Pro question.
        `options` is a dict like {'A': 'text', 'B': 'text', ...}
        """
        subject_formatted = subject.replace("_", " ").title() # MMLU-Pro subjects often use underscores
        prompt = f"The following is a multiple choice question about {subject_formatted}.\n\n"
        prompt += f"Question: {question_text}\n"
        for i in range(10): # Supports A-J options
            option_letter = chr(ord('A') + i)
            if option_letter in options and options[option_letter] is not None:
                prompt += f"{option_letter}. {options[option_letter]}\n"
        prompt += "Provide your final answer in JSON format as {\"Answer\": \"SELECTED_LETTER\"}, where SELECTED_LETTER is the letter of your chosen option."
        
        return prompt

    def _load_data(self):
        loaded_questions = []
        logging.info(f"Loading data for {self.HF_DATASET_NAME} (split: {self.data_split}, config: 'default')...")

        try:
            # TIGER-Lab/MMLU-Pro uses a "default" configuration containing all subjects.
            full_dataset = load_dataset(self.HF_DATASET_NAME, name="default", split=self.data_split)
            logging.info(f"Full dataset (default config, {self.data_split} split) loaded with {len(full_dataset)} items.")

            if not full_dataset or len(full_dataset) == 0:
                logging.error(f"ERROR: Dataset {self.HF_DATASET_NAME} (default/{self.data_split}) is empty or failed to load.")
                return []

            items_by_subject = {}
            for item in full_dataset:
                subject_val = item.get(self.COL_CATEGORY)
                if subject_val is None:
                    logging.warning(f"WARNING: Item missing category field ('{self.COL_CATEGORY}'). Item: {str(item)[:100]}")
                    subject_val = "unknown_subject" # Group items without a category

                # Apply subject filtering if specified by the user
                if self.subjects_to_run_filter and subject_val not in self.subjects_to_run_filter:
                    continue # Skip this item if its subject is not in the desired list

                if subject_val not in items_by_subject:
                    items_by_subject[subject_val] = []
                items_by_subject[subject_val].append(item)
            
            if self.subjects_to_run_filter:
                logging.info(f"Filtered down to {len(items_by_subject)} subjects based on input filter.")


            for subject_name, items_in_subject in items_by_subject.items():
                current_subject_items_to_load = items_in_subject
                
                num_questions_total = len(items_in_subject)

                if self.percentage_per_subject is not None:
                    percentage = max(0.0, min(100.0, self.percentage_per_subject))
                    num_to_take = math.ceil(num_questions_total * (percentage / 100.0))
                    logger.info(f"Limiting to {percentage}% ({num_to_take} of {num_questions_total}) questions for subject '{subject_name}'.")
                    current_subject_items_to_load = items_in_subject[:num_to_take]
                else:
                    logger.info(f"No percentage Limit set, Loading all {num_questions_total} questions for subject '{subject_name}'.")
                    percentage_per_subject = 100.0
                # <<< END: Updated logic >>>

                logging.debug(f"Processing {len(current_subject_items_to_load)} questions for subject '{subject_name}'...")
                for item_idx, item in enumerate(current_subject_items_to_load):
                    options_dict = {}
                    choices_list_from_item = item.get(self.COL_CHOICES_LIST)

                    # The 'options' field in TIGER-Lab/MMLU-Pro on HF is directly a list of strings.
                    # If loading from a CSV where it might be a string-representation-of-a-list, ast.literal_eval would be needed.
                    # For direct HF `load_dataset`, it should already be a Python list.
                    if isinstance(choices_list_from_item, list):
                        for i, choice_text in enumerate(choices_list_from_item):
                            if i < 10: # Supports A-J
                                option_letter = chr(ord('A') + i)
                                options_dict[option_letter] = str(choice_text)
                            else:
                                break # MMLU-Pro shouldn't exceed J
                    elif isinstance(choices_list_from_item, str): # Fallback for stringified list (less likely from load_dataset)
                        try:
                            parsed_list = ast.literal_eval(choices_list_from_item)
                            if isinstance(parsed_list, list):
                                for i, choice_text in enumerate(parsed_list):
                                    if i < 10: option_letter = chr(ord('A') + i); options_dict[option_letter] = str(choice_text)
                                    else: break
                            else:
                                logging.warning(f"Parsed '{self.COL_CHOICES_LIST}' is not a list for item {item.get(self.COL_QUESTION_ID, item_idx)} in {subject_name}.")
                        except (ValueError, SyntaxError):
                             logging.error(f"Could not parse string choices_list for item {item.get(self.COL_QUESTION_ID, item_idx)} in {subject_name}.")
                    else:
                        logging.warning(f"Choices list ('{self.COL_CHOICES_LIST}') is not a list or parsable string for item {item.get(self.COL_QUESTION_ID, item_idx)} in {subject_name}. Options will be empty.")

                    # Ensure all A-J keys exist, defaulting to None if not populated from choices_list
                    for i in range(10): # A-J
                        letter = chr(ord('A') + i)
                        if letter not in options_dict:
                            options_dict[letter] = None
                    
                    valid_options_for_prompt = {k: v for k, v in options_dict.items() if v is not None}
                    
                    question_text_val = str(item.get(self.COL_QUESTION_TEXT, ''))
                    
                    # Determine correct answer letter: Prefer 'answer' field, fallback to 'answer_index'
                    correct_answer_char = "INVALID_ANSWER" # Default
                    answer_letter_from_item = item.get(self.COL_ANSWER_LETTER)
                    answer_index_from_item = item.get(self.COL_ANSWER_INDEX)

                    if answer_letter_from_item is not None:
                        correct_answer_char = str(answer_letter_from_item).strip().upper()
                    elif answer_index_from_item is not None:
                        try:
                            ans_idx = int(answer_index_from_item)
                            if isinstance(choices_list_from_item, list) and 0 <= ans_idx < len(choices_list_from_item) and ans_idx < 10:
                                correct_answer_char = chr(ord('A') + ans_idx)
                            else:
                                logging.warning(f"Answer index {ans_idx} out of bounds or choices list not available for item {item.get(self.COL_QUESTION_ID, item_idx)} in {subject_name}.")
                        except ValueError:
                            logging.error(f"Answer index '{answer_index_from_item}' is not a valid integer for item {item.get(self.COL_QUESTION_ID, item_idx)} in {subject_name}.")
                    else:
                        logging.warning(f"No answer letter or index found for item {item.get(self.COL_QUESTION_ID, item_idx)} in {subject_name}.")


                    question_id_val = str(item.get(self.COL_QUESTION_ID, f"genid_{item_idx}"))

                    question_data = {
                        "id": f"{subject_name}_{question_id_val}",
                        "subject": subject_name,
                        "question_text": question_text_val,
                        "options": options_dict, # The A-J dict
                        "correct_answer_char": correct_answer_char,
                        "prompt": self._format_prompt(subject_name, question_text_val, valid_options_for_prompt)
                    }
                    loaded_questions.append(question_data)

        except Exception as e:
            logging.error(f"Major failure during MMLU-Pro data loading or processing: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for better debugging

        logging.info(f"Total MMLU-Pro questions loaded: {len(loaded_questions)}")
        return loaded_questions

    def get_questions(self):
        if not self.questions: # Load data only once
            self.questions = self._load_data()
        return self.questions

    def _extract_choice(self, model_response: str) -> str | None:
        if not model_response:
            return None
        
        # 1. Preprocessing: Remove <think>...</think> blocks
        processed_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL | re.IGNORECASE)
        processed_response = processed_response.strip()

        if not processed_response:
            return None

        # 2. Primary Extraction: Attempt to parse as a complete, valid JSON object. We expect something like {"Answer": "X"}
        try:
            start_brace = processed_response.find('{')
            end_brace = processed_response.rfind('}')
            
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str_candidate = processed_response[start_brace : end_brace + 1]
                parsed_json = json.loads(json_str_candidate)
                if isinstance(parsed_json, dict):
                    # Look for "Answer" or "answer" key
                    for key, value in parsed_json.items():
                        if key.lower() == "answer":
                            if isinstance(value, str) and len(value) == 1 and 'A' <= value.upper() <= 'J':
                                return value.upper()
                            break # Found the key, no need to check others
        except (json.JSONDecodeError, Exception):
            # If JSON parsing fails, pass silently to the next fallback method.
            logger.debug("Full JSON parsing failed, attempting fallback methods.Attempted reponse: {processed_response}")
            pass
        
        # 3. Fallback A: Targeted regex for malformed/incomplete JSON.
        # This looks specifically for the "Answer": "X" pattern.
        match = re.search(r'["\']Answer["\']\s*:\s*["\']([A-J])["\']', processed_response, re.IGNORECASE)
        if match:
            logger.debug(f"Extracted '{match.group(1).upper()}' using targeted JSON regex fallback.")
            return match.group(1).upper()

        # 4. Fallback B: Regex for natural language answers
        match = re.search(r"(?:correct|answer|option)\s+(?:is|was)\s*:?\s*\(?([A-J])\)?", processed_response, re.IGNORECASE)
        if match:
            logger.debug(f"Extracted '{match.group(1).upper()}' using natural language regex.")
            return match.group(1).upper()

        match = re.match(r"\s*([A-J])(?:[.)\s]|$)", processed_response)
        if match:
            logger.debug(f"Extracted '{match.group(1).upper()}' using start-of-string choice regex.")
            return match.group(1).upper()
            
        logger.warning(f"Could not extract a valid choice from response: '{processed_response[:100]}...'")
        return None

    def evaluate(self, model_response: str, question_data: dict) -> (float | None):
        extracted_choice = self._extract_choice(model_response)
        correct_answer = question_data.get("correct_answer_char")

        # Minimal debug for incorrect/failed extractions if needed
        # if extracted_choice != correct_answer or correct_answer == "INVALID_ANSWER":
        #     logging.info(f"Debug Eval: ID={question_data['id']}, Expected='{correct_answer}', Got='{extracted_choice}', Response='{model_response[:70]}...'")
        logging.info(f"Evaluating question {question_data['id']}: Expected '{correct_answer}', Got '{extracted_choice}'")
        if correct_answer == "INVALID_ANSWER": return 0.0
        if extracted_choice is None: return 0.0
        if extracted_choice == correct_answer: return 1.0
        return 0.0