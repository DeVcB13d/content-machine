# content-machine

## Installation

To run, preferably on ```python 3.10```

1. Get openai API key : https://platform.openai.com/api-keys

2. Create a .env file and set ```OPENAI_API_KEY = "<api_key> sk-"```

3. Install requirements : ```pip install -r requirements.txt```

4. Convert the file into vectors : ```python build_rag.py <file_path>.pdf```

5. Run the RAG model : ```python rag.py```