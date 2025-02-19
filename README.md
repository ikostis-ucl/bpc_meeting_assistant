# Meeting Minutes Assistant

A Python-based question answering system that processes and analyzes meeting minutes documents, providing context-aware responses with temporal tracking and document source references.

## Problem Statement

[To be filled in]

## Features

- **Document Processing & Indexing**
  - PDF document parsing with OCR capabilities using EasyOCR
  - Automated metadata extraction (dates, participants)
  - Vector storage using LlamaIndex
  - Support for temporal document organization
  - French language support

- **Query Processing**
  - Natural language query understanding in French
  - Time-aware document retrieval with customizable time steps
  - Context-sensitive answer generation using Groq LLM
  - Document source citations with page references
  - Answer similarity detection to prevent redundancy

- **User Interface**
  - Interactive Gradio-based GUI with chat interface
  - PDF document preview with source highlighting
  - Temporal navigation of responses
  - Predefined example queries
  - Support for both interactive and benchmark modes

## Architecture

### Core Components

1. **Data Processing (`app/engine/data_processing/`)**
   - `Storage`: Handles document parsing, OCR, and vector storage
   - `DataLoaders`: Manages index loading and timestamp tracking
   - `MetadataExtractors`: Extracts keywords and involved parties

2. **Inference Engine (`app/engine/inference/`)**
   - `BaseInference`: Core temporal query functionality
   - `GroqInference`: Implementation using Groq's LLM
   - `Judge`: Answer similarity evaluation system

3. **User Interface (`app/engine/gui/`)**
   - `GUI`: Gradio interface with chat and document preview
   - Support for multiple users in production mode
   - Terminal output formatting and progress indicators

4. **Utilities (`app/utils/`)**
   - Console output formatting
   - Data processing helpers
   - Inference utilities

### External Dependencies

- **LLM Services**
  - Groq API for answer generation
  - LlamaParse for document parsing
  - HuggingFace for embeddings (Solon)
  - ColBERT for answer reranking

## Setup

### Prerequisites
- Python 3.x
- Git
- pip package manager

### Installation

1. Clone the repository
```bash
git clone [repository-url]
cd meeting-minutes-assistant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create environment file
Create `app/assets/.env/.env` with the following API keys:
```
LLAMA_PARSE_KEY=llx-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
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
- `input_path`: Path to input documents
- `storage_dir`: Vector database location
- `embeddings_model`: HuggingFace embedding model
- `time_freq`: Time step for analysis (months)
- API credentials and paths

## Production Deployment

For production deployment, set the following environment variables:
- `GRADIO_AUTH_PAIRS`: Comma-separated user:password pairs
- `SERVER_NAME`: Server hostname
- `SSL_KEYFILE`: Path to SSL key
- `SSL_CERTFILE`: Path to SSL certificate

Run with production flag:
```bash
python run_gui.py --prod
```
