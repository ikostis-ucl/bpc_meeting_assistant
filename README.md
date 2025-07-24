# Meeting Minutes Assistant

A Python-based question answering system that processes and analyzes meeting minutes documents, providing context-aware responses with temporal tracking and document source references.

## Problem Statement

In large-scale construction projects, decisions evolve, details are revised, and older decisions are replaced. All these changes are documented in the meeting minutes held throughout the project's duration. Tracking and storing this evolving information can be a tedious task for project managers and other professionals who need to consult these documents.

To address this challenge, we propose an application designed to streamline access to this knowledge for a specific project. The solution is a conversational agent (or chatbot) based on Large Language Models and Retrieval-Augmented Generation. Its purpose is to preserve the timeline of information and retrieve relevant pieces of it at specific points in time. This is achieved using a custom information retrieval algorithm that considers both the semantic similarity of a user's question and the date when the decision was made.

## Features

- **Document Processing & Indexing**
  - PDF document parsing via LlamaParse.
  - Automated metadata extraction (dates, participants).
  - Vector storage using LlamaIndex.
  - Support for temporal document organization.
  - French language support.

- **Query Processing**
  - Natural language query understanding in French.
  - Time-aware document retrieval with customizable time steps.
  - Context-sensitive answer generation using Groq LLM.
  - Document source citations with page references.
  - Answer similarity detection to prevent redundancy.

- **User Interface**
  - Interactive Gradio-based GUI with chat interface.
  - PDF document preview with source highlighting.
  - Temporal navigation of responses.
  - Predefined example queries.
  - Support for both interactive and benchmark modes.

## Architecture

### Core Components

1. **Data Processing (`app/engine/data_processing/`)**
   - `Storage`: Handles document parsing, metadata extraction, and vector storage.
   - `DataLoaders`: Manages index loading and timestamp tracking.
   - `MetadataExtractors`: Extracts keywords and involved parties.

2. **Inference Engine (`app/engine/inference/`)**
   - `BaseInference`: Core temporal query functionality.
   - `GroqInference`: Implementation using Groq's LLM.
   - `Judge`: Answer similarity evaluation system.

3. **User Interface (`app/engine/gui/`)**
   - `GUI`: Gradio interface with chat and document preview.
   - Support for multiple users in production mode.
   - Terminal output formatting and progress indicators.

4. **Utilities (`app/utils/`)**
   - Console output formatting.
   - Data processing helpers.
   - Inference utilities.

### External Dependencies

- **LLM Services**
  - Groq API for answer generation.
  - LlamaParse for document parsing.
  - HuggingFace for embeddings.
  - ColbertRerank for answer reranking.

## Setup

### Prerequisites
- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bpc-meeting-assistant
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   poetry lock
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```


4. **Create environment file:**

    Create `app/assets/.env/.env` with the following API keys:
    ```
    LLAMA_PARSE_KEY=llx-...
    OPENAI_API_KEY=sk-...
    GROQ_API_KEY=gsk_...
    ```

### Usage

1. **Index Documents**
    ```bash
    python run_kb_indexing.py
    ```

2. **Run Interactive GUI**
    ```bash
    python run_gui.py
    ```

3. **Run Benchmark Tests**
    ```bash
    python run_inference.py
    ```

## Directory Structure

```
meeting-minutes-assistant/
├── app/
│   ├── assets/            # Assets and environment files
│   ├── engine/           
│   │   ├── data_processing/  # Document processing
│   │   ├── gui/             # User interface
│   │   └── inference/       # Query processing
│   ├── scripts/          # Main execution scripts
│   └── utils/            # Utility functions
├── data/
│   ├── input/            # Input documents
│   ├── tests/            # Test documents
│   └── vector_db/        # Vector storage
└── resources/            # Model cache and resources
```

## Configuration

Key parameters in `configuration.py`:

### Data Processing
- `input_path`: Path to input documents (default: `./data/input`)
- `storage_dir`: Vector database location (default: `./data/vector_db`)

### Embeddings
- `embeddings_model`: HuggingFace embedding model (default: `OrdalieTech/Solon-embeddings-large-0.1`)
- `embeddings_cache_dir`: Embeddings cache directory (default: `./resources/.emb_models`)

### LLM Models (Groq)
- `groq_model_inference`: Main inference model (default: `llama-3.3-70b-versatile`)
- `groq_model_inference_judge`: Model for answer similarity assessment (default: `llama-3.3-70b-versatile`)
- `groq_model_indexing_extraction`: Model for information extraction during indexing (default: `llama-3.3-70b-versatile`)
- `groq_model_indexing_kw`: Model for keyword extraction (default: `llama-3.1-8b-instant`)
- `groq_model_inference_tpm`: Tokens per minute limit (default: 12000)

### Retrieval Settings
- `time_freq`: Time step duration in months for temporal analysis (default: 5)
- `cutoff_percentage`: Percentage of retrieved nodes for context (default: 0.05)

## Production Deployment

For production deployment, set the following environment variables:
- `GRADIO_AUTH_PAIRS`: Comma-separated user:password pairs

Run with production flag:
```bash
python run_gui.py --prod
```
