# LLM Module: A Unified LLM Client Architecture

**Version**: 2.0.0

## Overview

The `llm/` directory provides a robust, object-oriented architecture for interacting with various Large Language Models (LLMs). It uses a factory pattern to decouple client creation from usage and an abstract base class to ensure a consistent interface across all supported LLM providers. This design makes it easy to switch between LLMs (like Gemini, LMStudio, and Ollama) and to extend the system with new providers in the future.

## Core Architectural Patterns

This module is built on two primary design patterns:

1.  **Abstract Base Class (Interface)**: `base_client.py` defines an abstract class `BaseLLMClient`. It serves as a contract, declaring the methods (`generate`, `is_available`) that all concrete LLM clients must implement. This enables polymorphism, allowing different clients to be used interchangeably.

2.  **Factory Pattern**: `client_factory.py` implements the `LLMClientFactory`. This factory is the designated entry point for creating LLM client instances. It abstracts away the instantiation logic, making it simple to request a client for a specific provider (e.g., `"gemini"`) without needing to know the details of its creation.

## Key Components

| File | Description |
| :--- | :--- |
| **`client_factory.py`** | **(Entry Point)** The factory for creating LLM clients. Use this to get instances of any supported LLM provider. |
| **`base_client.py`** | The abstract base class that defines the common interface for all LLM clients. |
| **`gemini_client.py`** | The concrete client for Google's Gemini API. Handles Gemini-specific message formatting and API calls. |
| **`lmstudio_client.py`** | The concrete client for LM Studio, which provides an OpenAI-compatible local server. |
| **`ollama_client.py`** | The concrete client for the Ollama local LLM server. |
| **`chat_handler.py`** | A utility module for preparing messages. It loads the system prompt, injects RAG context, and formats the conversation history into the list of messages required by the clients. |
| **`config_loader.py`** | A powerful, centralized configuration manager. It loads settings from `config/app.yaml`, resolves paths, and reads environment variables and secrets. |
| `chat_styles.css` | CSS styles used by the Streamlit frontend for rendering the chat interface. |

## Usage Examples

### Recommended Workflow: Factory + Chat Handler

This example shows the complete, recommended workflow for generating a response in a RAG application.

```python
from llm.client_factory import LLMClientFactory
from llm.chat_handler import build_messages

# 1. Select your desired LLM provider
provider = "gemini"  # or "lmstudio", "ollama"

# 2. Use the factory to create a client instance
# The factory handles loading all necessary configurations automatically.
try:
    llm_client = LLMClientFactory.create_from_string(provider)
except ValueError as e:
    print(e)
    # Handle error, maybe fall back to a default client
    exit()

# 3. (In a RAG system) Fetch context from your retrieval pipeline
retrieved_context = "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources."

# 4. Use the chat handler to build the message list
# This injects the system prompt and context correctly.
user_query = "What is RAG?"
chat_history = [
    {"role": "user", "content": "Hi, who are you?"},
    {"role": "assistant", "content": "I am an AI assistant."}
]
messages = build_messages(
    query=user_query,
    context=retrieved_context,
    history=chat_history
)

# 5. Generate a response
if llm_client.is_available():
    try:
        response = llm_client.generate(messages)
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"An error occurred during generation: {e}")
else:
    print(f"The selected LLM provider '{provider}' is not available. Please check your configuration.")

```

### Direct Client Creation (Advanced)

While using the factory is recommended, you can also instantiate clients directly if you need to provide specific, ad-hoc configurations.

```python
from llm.gemini_client import GeminiClient

# Custom configuration overrides what's in app.yaml
custom_config = {
    "model": "gemini-1.5-pro-latest",
    "temperature": 0.9,
}

gemini = GeminiClient(config=custom_config)

# Generate a response
if gemini.is_available():
    response = gemini.generate([{"role": "user", "content": "Hello, world!"}])
    print(response)
```

## Configuration

All LLM settings are centralized in `config/app.yaml` under the `llm:` key. The `config_loader.py` module reads this file and provides functions to resolve the final settings, incorporating environment variables and Streamlit secrets where applicable.

**Example `config/app.yaml` structure:**
```yaml
llm:
  gemini:
    model: "gemini-1.5-flash"
    temperature: 0.7
    # API key is resolved from secrets or environment variables
  lmstudio:
    base_url: "http://localhost:1234/v1"
    model: "loaded-local-model-name"
    # ...
  ollama:
    base_url: "http://localhost:11434"
    model: "gemma:latest"
    # ...
```

The loader will prioritize settings in this order:
1.  Explicit arguments passed to a client's constructor.
2.  Environment variables (e.g., `GOOGLE_API_KEY`, `OLLAMA_BASE_URL`).
3.  Streamlit secrets (in `secrets.toml`).
4.  Values from `config/app.yaml`.

## How to Add a New LLM Provider

Thanks to the architecture, adding a new provider is straightforward:

1.  **Create a New Client**: Create a new file (e.g., `my_new_client.py`) in the `llm/` directory.
2.  **Implement the Interface**: Inside the new file, create a class that inherits from `BaseLLMClient` and implements the `generate()` and `is_available()` methods.
3.  **Update the Factory**:
    *   Add your new client to the `LLMProvider` enum in `client_factory.py`.
    *   Update the `create()` method in `LLMClientFactory` to handle the new provider enum.
4.  **Add Configuration**: Add a new section for your client's settings in `config/app.yaml`.
5.  **Update Config Loader**: Add a `resolve_my_new_settings()` function to `config_loader.py` to load your client's configuration.

That's it! Your new client is now integrated and available throughout the application via the factory.