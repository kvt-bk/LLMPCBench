import requests
import json
import logging

logger = logging.getLogger(__name__)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

def check_ollama_connection():
    """Checks if the Ollama API is running and reachable."""
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        response.raise_for_status()
        logger.info("Ollama API connection successful.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API is not reachable at http://localhost:11434. Please ensure Ollama is running. If you are using a different port, please update the OLLAMA_API_URL variable in olalma_client.py")
        logger.error(f"Error details: {e}")
        return False

def get_ollama_response(model_name: str, prompt: str, options: dict = {}):
    """
    Sends a prompt to the Ollama API and gets a response.

    Args:
        model_name (str): The name of the Ollama model to use.
        prompt (str): The prompt to send to the model.

    Returns:
        tuple: (generated_text, tokens_per_second, error_message)
               tokens_per_second is None if an error occurs or if metrics are unavailable.
               error_message is None if successful.
    """
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  
            #"format": "json",  # Request JSON response. This is confusing some models, so not using it for now.
            "options": options,
            "system": "You are an expert AI assistant that excels at following user instructions to answer questions accurately."

        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) 
        response.raise_for_status()  # Raise an exception for HTTP errors
        logger.debug(f"Ollama API Response: {response.text}")
        response_data = response.json()
        generated_text = response_data.get("response", "{}").strip()

        # Calculate tokens per second
        # eval_count = number of tokens in the response
        # eval_duration = nanoseconds for generating the response
        eval_count = response_data.get("eval_count")
        eval_duration_ns = response_data.get("eval_duration")

        tokens_per_second = None
        if eval_count is not None and eval_duration_ns is not None and eval_duration_ns > 0:
            eval_duration_s = eval_duration_ns / 1_000_000_000  # Convert nanoseconds to seconds
            tokens_per_second = eval_count / eval_duration_s
        
        return generated_text, tokens_per_second, None

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return None, None, f"API request failed: {e}"
    except json.JSONDecodeError:
        logger.error("Failed to decode Ollama API response.")
        return None, None, "Failed to decode API response."
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_ollama_response: {e}")
        return None, None, f"An unexpected error occurred: {e}"

def list_ollama_models():
    """
    Lists locally available Ollama models.
    Note: Ollama's primary API for listing models is via /api/tags.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models_data = response.json()
        return [model['name'] for model in models_data.get('models', [])], None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch models: {e}")
        return [], f"Failed to fetch models: {e}"
    except Exception as e:
        return [], f"An unexpected error occurred while fetching models: {e}"

if __name__ == '__main__':
    # Test the client
    logger.info("Available Ollama models:")
    models, error = list_ollama_models()
    if error:
        logger.error(f"Error: {error}")
    elif models:
        for model in models:
            logger.info(f"- {model}")
        
        # Test generation with the first available model if any
        if models:
            test_model = models[0] # Use the first model found
            logger.info(f"\nTesting generation with model: {test_model}")
            prompt = "Why is the sky blue?"
            text, tps, err = get_ollama_response(test_model, prompt)
            if err:
                logger.error(f"Error: {err}")
            else:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Response: {text}")
                if tps is not None:
                    logger.info(f"Performance: {tps:.2f} tokens/second")
                else:
                    logger.warning("Performance metrics not available.")
    else:
        logger.warning("No Ollama models found or Ollama is not running.")