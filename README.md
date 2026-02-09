# J.A.R.V.I.S - Just A Rather Very Intelligent System

An intelligent AI assistant built with FastAPI, LangChain, and Groq AI. JARVIS provides two modes of interaction: General Chat (pure LLM, no web search) and Realtime Chat (with Tavily web search). The system learns from user data files and past conversations, maintaining context across sessions.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Operating System**: Windows, macOS, or Linux (fully cross-platform)
- **API Keys** (set in `.env` file):
  - `GROQ_API_KEY` - Get from <https://console.groq.com> (required). You can add more keys for round-robin and fallback (see [Multiple Groq API keys](#multiple-groq-api-keys)).
  - `TAVILY_API_KEY` - Get from <https://tavily.com> (optional, for realtime mode)
  - `GROQ_MODEL` - Optional, defaults to "llama-3.3-70b-versatile"

### Installation

1. **Clone/Download** this repository

2. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

1. **Create `.env` file** in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
# Optional: add more keys for round-robin and fallback (GROQ_API_KEY_2, GROQ_API_KEY_3, ...)
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
# Optional: assistant name (default: Jarvis). Tone and personality stay the same.
# ASSISTANT_NAME=Jarvis
# Optional: how to address the user; otherwise uses learning data/chats.
# JARVIS_USER_TITLE=Sir
```

1. **Start the server**:

```bash
python run.py
```

The server will start at **<http://localhost:8000>**

1. **Test the system** (in another terminal):

```bash
python test.py
```

## 📋 Features

### Core Features

- ✅ **Dual Chat Modes**: General chat (pure LLM, no web search) and Realtime chat (with Tavily search)
- ✅ **Session Management**: Conversations persist across server restarts
- ✅ **Learning System**: Learns from user data files and past conversations via semantic search (no token limit blow-up). No hardcoded names—assistant name and user title come from `ASSISTANT_NAME` and `JARVIS_USER_TITLE` in `.env`, or from learning data and chats.
- ✅ **Learning data on restart**: Add or edit `.txt` files in `database/learning_data/` and restart the server to pick them up
- ✅ **Vector Store**: FAISS index of learning data + past chats; only relevant chunks are sent to the LLM so you never hit token limits
- ✅ **Assistant Personality**: Sophisticated, witty, professional tone with British humor (name configurable via `ASSISTANT_NAME` in `.env`)

### Technical Features

- **Learning data**: All `.txt` files in `database/learning_data/` are indexed in the vector store. The AI answers from this data by **retrieving relevant chunks** per question (not by sending all text in every prompt), so you can add many files without exceeding token limits.
- **Hot-reload**: A background check runs every 15 seconds. If any `.txt` in `learning_data/` is new or modified, the vector store is rebuilt so new content is learned instantly.
- **Curly Brace Escaping**: Prevents LangChain template variable errors
- **Smart Response Length**: Adapts answer length based on question complexity
- **Clean Formatting**: No markdown, asterisks, or emojis in responses
- **Time Awareness**: AI knows current date and time

## 🏗️ Architecture

### System Overview

```
User Input
    ↓
FastAPI Endpoints (/chat or /chat/realtime)
    ↓
ChatService (Session Management)
    ↓
GroqService or RealtimeGroqService
    ↓
VectorStoreService (Context Retrieval)
    ↓
Groq AI (LLM Response Generation)
```

### Component Breakdown

1. **FastAPI Application** (`app/main.py`)
   - REST API endpoints
   - CORS middleware
   - Application lifespan management

2. **Chat Service** (`app/services/chat_service.py`)
   - Session creation and management
   - Message storage (in-memory and disk)
   - Conversation history formatting

3. **Groq Service** (`app/services/groq_service.py`)
   - General chat mode (pure LLM, no web search)
   - Retrieves relevant context from vector store (learning data + past chats) per request; no full-text dump, so token usage stays bounded

4. **Realtime Service** (`app/services/realtime_service.py`)
   - Extends GroqService
   - Adds Tavily web search
   - Combines search results with AI knowledge

5. **Vector Store Service** (`app/services/vector_store.py`)
   - FAISS vector database
   - Embeddings generation (HuggingFace)
   - Semantic search for context retrieval

6. **Configuration** (`config.py`)
   - Centralized settings
   - User context loading
   - System prompt definition

## 📁 Project Structure

```
JARVIS/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application and API endpoints
│   ├── models.py               # Pydantic data models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py     # Session and conversation management
│   │   ├── groq_service.py      # General chat AI service
│   │   ├── realtime_service.py  # Realtime chat with web search
│   │   └── vector_store.py      # FAISS vector store and embeddings
│   └── utils/
│       ├── __init__.py
│       └── time_info.py         # Current date/time information
├── database/
│   ├── learning_data/           # User data files (.txt)
│   │   ├── userdata.txt        # Personal information (auto-loaded)
│   │   ├── system_context.txt  # System context (auto-loaded)
│   │   └── *.txt               # Any other .txt files (auto-loaded)
│   ├── chats_data/             # Saved conversations (.json)
│   └── vector_store/           # FAISS index files
├── config.py                   # Configuration and settings
├── run.py                      # Server startup script
├── test.py                     # CLI test interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔌 API Endpoints

### POST `/chat`

General chat endpoint (pure LLM, no web search).

**Request:**

```json
{
  "message": "What is Python?",
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "response": "Python is a high-level programming language...",
  "session_id": "session-id-here"
}
```

### POST `/chat/realtime`

Realtime chat endpoint (with Tavily web search).

**Request:**

```json
{
  "message": "What's the latest AI news?",
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "response": "Based on recent search results...",
  "session_id": "session-id-here"
}
```

### GET `/chat/history/{session_id}`

Get chat history for a session.

**Response:**

```json
{
  "session_id": "session-id",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Good day. How may I assist you?"}
  ]
}
```

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "vector_store": true,
  "groq_service": true,
  "realtime_service": true,
  "chat_service": true
}
```

### GET `/`

API information endpoint.

## 🧠 How It Works

### 1. Learning Data and Vector Store

- **At startup:** All `.txt` files in `database/learning_data/` and all past chats in `chats_data/` are loaded, chunked, embedded, and stored in a FAISS vector store.
- **Restart for new learning data:** Restart the server after adding or changing `.txt` files in `learning_data/`; the vector store is rebuilt on startup.
- **No full dump:** Learning data is never sent in full in the prompt. Only the top-k retrieved chunks (from learning data + past conversations) are sent per request, so token usage stays bounded.

### 2. Vector Store Creation

On startup (and when learning_data changes):

- Loads all `.txt` files from `learning_data/`
- Loads all past conversations from `chats_data/`
- Converts text to embeddings using HuggingFace model
- Creates FAISS index for fast similarity search
- Saves index to disk

### 3. Message Processing (General Mode)

1. User sends message via `/chat` endpoint
2. ChatService creates/retrieves session
3. User message stored in session
4. GroqService retrieves relevant context from the vector store:
   - Relevant chunks from learning data (`.txt` files) and past conversations (semantic search)
   - Current time information
5. System prompt built with all context
6. Groq AI generates response
7. Response stored in session
8. Session saved to disk

### 4. Message Processing (Realtime Mode)

1. User sends message via `/chat/realtime` endpoint
2. Same session management as general mode
3. RealtimeGroqService:
   - Searches Tavily for real-time information
   - Retrieves relevant context (same as general mode)
   - Combines search results with context
   - Generates response with current information
4. Response stored and saved

### 5. Context Retrieval

When answering a question:

- Vector store performs semantic search
- Finds most relevant documents (k=6 by default)
- Documents can be from:
  - Learning data files
  - Past conversations
- Context is escaped (curly braces) to prevent template errors
- Context added to system prompt

### 6. Session Management

- **Server-managed**: If no `session_id` provided, server generates UUID
- **User-managed**: If `session_id` provided, server uses it
- Sessions persist across server restarts (loaded from disk)
- Both `/chat` and `/chat/realtime` share the same session
- Sessions saved to `database/chats_data/` as JSON files

## 🎯 Usage Examples

### Using test.py (CLI Interface)

```bash
python test.py
```

**Commands:**

- `1` - Switch to General Chat mode
- `2` - Switch to Realtime Chat mode
- `/history` - View chat history
- `/clear` - Start new session
- `/quit` - Exit

### Using Python Requests

```python
import requests

# General chat
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What is machine learning?",
        "session_id": "my-session-id"
    }
)
print(response.json()["response"])

# Realtime chat
response = requests.post(
    "http://localhost:8000/chat/realtime",
    json={
        "message": "What's happening in AI today?",
        "session_id": "my-session-id"  # Same session continues
    }
)
print(response.json()["response"])
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env

# Required
GROQ_API_KEY=your_groq_api_key


# Optional: add more keys for round-robin and fallback when one hits rate limit
# GROQ_API_KEY_2=second_key
# GROQ_API_KEY_3=third_key


# Optional (for realtime mode)
TAVILY_API_KEY=your_tavily_api_key


# Optional (defaults to llama-3.3-70b-versatile)
GROQ_MODEL=llama-3.3-70b-versatile


# Optional: assistant name (default: Jarvis). Tone and personality stay the same.
# ASSISTANT_NAME=Jarvis


# Optional: how to address the user (e.g. "Sir", "Mr. Smith"). If not set, the AI uses
# only learning data and conversation history to address the user (no hardcoded names).
# JARVIS_USER_TITLE=
```

### Multiple Groq API keys

You can add **multiple Groq API keys** so the server uses **every key one-by-one** in rotation and falls back to the next key if one fails.

- **Round-robin (one-by-one):** The server uses each key in order: 1st request → 1st key, 2nd request → 2nd key, 3rd request → 3rd key, then back to the 1st key, and so on. Every key you give is used in turn; no key is skipped.
- **Fallback:** If the chosen key fails (e.g. rate limit 429 or any error), the server tries the next key, then the next, until one succeeds or all have been tried.

In your `.env`, set as many keys as you want using this pattern:

```env
GROQ_API_KEY=your_first_key
GROQ_API_KEY_2=your_second_key
GROQ_API_KEY_3=your_third_key
# Add more: GROQ_API_KEY_4, GROQ_API_KEY_5, ... (no upper limit)
```

Only `GROQ_API_KEY` is required. Add `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, etc. for extra keys. Each key has its own daily token limit on Groq’s free tier, so multiple keys give you more capacity. The code that does round-robin and fallback is in `app/services/groq_service.py` (see `_invoke_llm` and module docstring for line-by-line explanation).

### System Prompt Customization

Edit `config.py` to modify:

- Assistant personality and tone (the assistant name is set via `ASSISTANT_NAME` in `.env`)
- Response length guidelines
- Formatting rules
- General behavior guidelines

### User Data Files

Add any `.txt` files to `database/learning_data/`:

- Files are automatically detected and loaded
- Content is always included in system prompt
- Files are loaded in alphabetical order
- No need to modify code when adding new files

**Example files:**

- `userdata.txt` - Personal information
- `system_context.txt` - System context
- `usersinterest.txt` - User interests
- Any other `.txt` file you add

## 🛠️ Technologies Used

### Backend

- **FastAPI**: Modern Python web framework
- **LangChain**: LLM application framework
- **Groq AI**: Fast LLM inference (Llama 3.3 70B)
- **Tavily**: AI-optimized web search API
- **FAISS**: Vector similarity search
- **HuggingFace**: Embeddings model (sentence-transformers)
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### Data Storage

- **JSON Files**: Chat session storage
- **FAISS Index**: Vector embeddings storage
- **Text Files**: User learning data

## 📝 Key Features Explained

### Learning Data (restart to pick up new files)

- **Indexing**: All `.txt` files in `database/learning_data/` are indexed in the vector store (with past chats). The AI **retrieves only relevant chunks** per question, so token usage stays bounded and you can add many files without hitting limits.
- **Restart to pick up new files**: New or changed `.txt` files in `learning_data/` are loaded when you restart the server (vector store is rebuilt on startup).
- **No full dump**: The system does not send all learning data in every prompt; it uses semantic search to pull only what’s relevant, so you never hit the token limit.

### Curly Brace Escaping

The `escape_curly_braces()` function:

- Prevents LangChain from interpreting `{variable}` as template variables
- Escapes braces by doubling them: `{` → `{{`, `}` → `}}`
- Applied to all context before adding to system prompt

**Why this matters**: Prevents template variable errors when content contains curly braces.

### Vector Store

The vector store:

- Converts text to numerical embeddings
- Stores embeddings in FAISS index
- Enables fast similarity search
- Rebuilt on every startup (always current)

**Why this matters**: Allows JARVIS to find relevant information from past conversations and learning data.

### Session Persistence

Sessions:

- Stored in memory during active use
- Saved to disk after each message
- Loaded from disk on server restart
- Shared between general and realtime modes

**Why this matters**: Conversations continue seamlessly across server restarts.

## 🐛 Troubleshooting

### Server won't start

- Check that `GROQ_API_KEY` is set in `.env`
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check port 8000 is not in use

### "Cannot connect to backend"

- Make sure server is running: `python run.py`
- Check server is on `http://localhost:8000`
- Verify no firewall blocking the connection

### Vector store errors

- Ensure `database/` directories exist
- Check file permissions on `database/` directory
- Delete `database/vector_store/` to rebuild index

### Template variable errors

- Should be fixed by curly brace escaping
- Check for any unescaped `{` or `}` in learning data files
- Restart server after fixing

### Realtime mode not working

- Check `TAVILY_API_KEY` is set in `.env`
- Verify Tavily API key is valid
- Check internet connection

## 🔒 Security Notes

- Session IDs are validated to prevent path traversal (checks for both `/` and `\`)
- API keys stored in `.env` (not in code)
- CORS enabled for all origins (adjust for production)
- No authentication (add for production use)

## 🌐 Cross-Platform Compatibility

**This code is fully cross-platform and works on:**

- ✅ **Windows** (Windows 10/11)
- ✅ **macOS** (all versions)
- ✅ **Linux** (all distributions)

**Why it's cross-platform:**

- Uses `pathlib.Path` for all file paths (handles `/` vs `\` automatically)
- Explicit UTF-8 encoding for all file operations
- No hardcoded path separators
- No shell commands or platform-specific code
- Standard Python libraries only
- Session ID validation checks both `/` and `\` for security

**Tested on:**

- macOS (Darwin)
- Windows (should work - uses standard Python practices)
- Linux (should work - uses standard Python practices)

## 📚 Development

### Running in Development Mode

```bash
python run.py
```

Auto-reload is enabled, so code changes restart the server automatically.

### Testing

```bash
# CLI test interface
python test.py

# Or use curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Project Structure Philosophy

- **Separation of Concerns**: Each service handles one responsibility
- **Configuration Centralization**: All settings in `config.py`
- **Type Safety**: Pydantic models for validation
- **Documentation**: Comprehensive docstrings in all modules

## 👤 Developer

**J.A.R.V.I.S** was developed by **Awais Qasim**, an online educator, business minded programmer, and FinTech enthusiast, and Crypto Trader known for innovative methods.

## 📄 License

MIT

---

**Made with ❤️ for intelligent conversations**

**Start chatting:** `python run.py` then `python test.py`
