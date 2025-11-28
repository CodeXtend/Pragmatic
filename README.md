# TruthShield

A conversational agent for fact extraction and verification using LLM and search tools.

## Installation

### As a package (for use in other programs)

```bash
pip install -e .
```

### Standalone

```bash
pip install -r requirements.txt
```

## Usage

### As a package in your code

```python
from truthshield import FactExtracter

# Initialize the fact extracter
extracter = FactExtracter()

# Run a query
response = extracter.run("What is the capital of France?")
print(response)

# Clear conversation memory if needed
extracter.clear_memory()
```

### As a CLI tool

```bash
python FactExtracter.py
```

Or if installed as a package:

```bash
truthshield
```

## Configuration

Create a `.env` file with:

```env
LLM_MODEL=your_model_id
GEMINI_API_KEY=your_api_key
```

## API Reference

### FactExtracter

#### `__init__(model_id=None, api_key=None, max_steps=10)`

Initialize the FactExtracter agent.

- `model_id`: The LLM model ID (optional, reads from `LLM_MODEL` env variable)
- `api_key`: The API key (optional, reads from `GEMINI_API_KEY` env variable)
- `max_steps`: Maximum steps the agent can take (default: 10)

#### `run(user_input: str) -> str`

Process user input and return the agent's response.

#### `clear_memory()`

Clear the conversation memory.

#### `get_memory() -> List[str]`

Get the current conversation history.
