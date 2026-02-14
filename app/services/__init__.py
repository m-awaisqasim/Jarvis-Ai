"""
SERVICES PACKAGE
=================

Business logic lives here. The API layer (app.main) calls these services;
they don't handle HTTP, only chat flow, LLM calls, and data.

MODELS:
    chat_service
    groq_service
    realtime_service - Realtime chat: Tavily search first, then same as qroq (GroqServices) 
    vector_store 
"""