Installation:
 install dependencies:
    pip install -r requirements.txt
    install offline ollama
    and llama3 model
   once installed:
   execute command: ollama serve

Usage:
execute command: uvicorn main:app --reload
in other terminal terminal:
   execute command: python -m http.server 9000

visit http://localhost:9000/frontend/ from the browser ask questions
