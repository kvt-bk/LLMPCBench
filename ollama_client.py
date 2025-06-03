import requests
import json
import time

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def get_ollama_response(model_name: str, prompt: str):
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
            "stream": False  # Set to False to get the full response at once
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) # Increased timeout
        response.raise_for_status()  # Raise an exception for HTTP errors

        response_data = response.json()
        generated_text = response_data.get("response", "").strip()

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
        return None, None, f"API request failed: {e}"
    except json.JSONDecodeError:
        return None, None, "Failed to decode API response."
    except Exception as e:
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
        return [], f"Failed to fetch models: {e}"
    except Exception as e:
        return [], f"An unexpected error occurred while fetching models: {e}"

if __name__ == '__main__':
    # Test the client
    print("Available Ollama models:")
    models, error = list_ollama_models()
    if error:
        print(f"Error: {error}")
    elif models:
        for model in models:
            print(f"- {model}")
        
        # Test generation with the first available model if any
        if models:
            test_model = models[0] # Use the first model found
            print(f"\nTesting generation with model: {test_model}")
            prompt = "Why is the sky blue?"
            text, tps, err = get_ollama_response(test_model, prompt)
            if err:
                print(f"Error: {err}")
            else:
                print(f"Prompt: {prompt}")
                print(f"Response: {text}")
                if tps is not None:
                    print(f"Performance: {tps:.2f} tokens/second")
                else:
                    print("Performance metrics not available.")
    else:
        print("No Ollama models found or Ollama is not running.")