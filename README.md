# Deepsearch
Testing LangChain with Ollama (smollm model)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure Ollama is running with the smollm model:
   ```
   ollama pull smollm:latest
   ollama run smollm:latest
   ```

## Running Tests

### Simple test:
```
python test_ollama.py
```

### Using pytest:
```
pytest test_ollama_with_pytest.py -v
```
