import os
import json
import litellm
from dotenv import load_dotenv
from src.logger_config import logger

# Register model costs once at module level
for model_id, cost_info in {
    "nvidia_nim/nvidia/llama-3.3-70b-instruct": {"max_tokens": 4096, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
    "nvidia_nim/nvidia/nemotron-mini-4b-instruct": {"max_tokens": 4096, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
    "openrouter/mistralai/mistral-7b-instruct": {"max_tokens": 4096, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
}.items():
    litellm.model_cost[model_id] = cost_info

def track_usage(kwargs, response_obj, start_time, end_time):
    usage = getattr(response_obj, 'usage', {})
    model = kwargs["model"]
    logger.info(f"LLM Success | Model: {model} | Tokens: {usage} | Latency: {end_time - start_time:.2f}s")

litellm.success_callback = [track_usage, "langfuse"]
litellm.failure_callback = ["langfuse"]

def init_langfuse():
    """Call once at startup after load_dotenv()"""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    
    if public_key.startswith("pk-lf-eu-"):
        default_host = "https://eu.cloud.langfuse.com"
    else:
        default_host = "https://cloud.langfuse.com"
    
    os.environ["LANGFUSE_HOST"] = default_host
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")

def load_settings(filename="settings.json"):
    settings_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(settings_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Settings file not found: {settings_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in settings: {e}")
        raise

def load_api_keys():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", "..", ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    
    init_langfuse()
    
    os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
    os.environ["OR_SITE_URL"] = "http://localhost:3000"
    
    return os.environ["NVIDIA_NIM_API_KEY"], os.environ["OPENROUTER_API_KEY"]

def get_providers(settings):
    configs = [
        ("use_nvidia", "nvidia_model", "nvidia_nim", "NVIDIA_LLAMA", "nvidia_cost_1k"),
        ("use_mistral_nvidia", "mistral_model", "nvidia_nim", "NVIDIA_MISTRAL", "mistral_cost_1k"),
        ("use_nemotron_nvidia", "nemotron_model", "nvidia_nim", "NVIDIA_NEMOTRON", "nemotron_cost_1k"),
        ("use_openrouter", "openrouter_model", "openrouter", "OPENROUTER", "openrouter_cost_1k"),
    ]
    
    providers = []
    for flag, model_key, prefix, label, cost_key in configs:
        if settings.get(flag):
            providers.append((settings[model_key], prefix, label, settings.get(cost_key, 0.0)))
    
    return providers
