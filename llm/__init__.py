"""
LLM Package - LLM Client Abstraction Layer
===========================================
OOP-based LLM clients with factory pattern and provider abstraction

Architecture:
- base_client.py: Abstract base class (BaseLLMClient)
- gemini_client.py: Google Gemini implementation
- lmstudio_client.py: LMStudio OpenAI-compatible implementation
- client_factory.py: Factory for creating clients
- chat_handler.py: Message formatting utilities
- config_loader.py: Configuration management

Usage:
    from llm.client_factory import LLMClientFactory
    
    # Create client
    client = LLMClientFactory.create_gemini()
    
    # Generate response
    messages = [{"role": "user", "content": "Hello"}]
    response = client.generate(messages)
"""
__version__ = "2.0.0"
