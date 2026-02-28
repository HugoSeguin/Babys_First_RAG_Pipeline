# 🧠 Baby's First RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline that lets you query across multiple documents using semantic search and a Gemini LLM backend.

---

## What It Does

This project lets you ask natural language questions and get answers grounded in your own documents — not just the model's training data.

It works in three steps:
1. **Load** text documents and split them into chunks
2. **Embed** those chunks into a FAISS vector store using HuggingFace sentence transformers
3. **Retrieve** the most relevant chunks for a query and pass them to Gemini to generate an answer

---

## Data Sources

The pipeline is pre-loaded with three documents:

| File | Description |
|---|---|
| `wiki_game_awards_2025.txt` | Wikipedia article on The Game Awards 2025 |
| `wiki_98th_oscars.txt` | Wikipedia article on the 98th Academy Awards |
| `chiirl_events.txt` | Chicago tech/innovation meetup events (Dec 2025) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `LangChain` | Text splitting and document management |
| `HuggingFace` (`all-mpnet-base-v2`) | Sentence embeddings |
| `FAISS` | Local vector similarity search |
| `Google Gemini` (`gemini-2.5-flash`) | LLM for answer generation |

---

## Setup

**1. Clone the repo:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key:**

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_api_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com)

**5. Add your data files** to the root directory:
- `wiki_game_awards_2025.txt`
- `wiki_98th_oscars.txt`
- `chiirl_events.txt`

---

## Usage

```bash
python rag_pipeline.py
```

To change the query, edit the `prompt` variable in `main()`:
```python
prompt = 'who performed at the 2025 game awards'
```

### Example Queries
```python
'Clair Obscur: Expedition 33 how many nominations at 2025 game awards'
'chicago tech meetup events dec 2025'
'time and location of 2026 academy awards'
```

---

## Project Structure

```
├── rag_pipeline.py         # Main pipeline script
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore
├── README.md
├── wiki_game_awards_2025.txt
├── wiki_98th_oscars.txt
└── chiirl_events.txt
```

---

## How RAG Works (Quick Explainer)

```
Your Question
     │
     ▼
Embed question → search FAISS vector store
     │
     ▼
Retrieve top-k most similar chunks
     │
     ▼
Inject chunks into prompt → send to Gemini
     │
     ▼
Grounded Answer ✓
```

Without RAG, the model only knows what it was trained on. With RAG, it can answer questions about *your* documents.

---

## License

MIT
