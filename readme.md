# South of France Travel Planning System

An intelligent document processing system that extracts structured information from travel guide PDFs and provides personalized recommendations for trip planning using semantic analysis and machine learning.

## 🎯 Overview

This system processes multiple travel guide PDFs about the South of France and creates intelligent, context-aware travel recommendations. It combines PDF text extraction, semantic chunking, and similarity matching to provide the most relevant information for specific travel planning needs.

## ✨ Features

- **PDF Structure Extraction**: Automatically detects headings, titles, and document structure from travel guide PDFs
- **Semantic Text Chunking**: Intelligently breaks down content into meaningful sections using sentence embeddings
- **Personalized Recommendations**: Matches content to specific personas and travel requirements
- **Multi-Document Analysis**: Processes multiple related documents and finds cross-document insights
- **Local Model Support**: Uses locally stored sentence transformer models for offline operation

## 📁 Project Structure

```
ah-21b/
├── PDFs/                           # Input travel guide PDFs
│   ├── South of France - Cities.pdf
│   ├── South of France - Cuisine.pdf
│   ├── South of France - History.pdf
│   ├── South of France - Restaurants and Hotels.pdf
│   ├── South of France - Things to Do.pdf
│   ├── South of France - Tips and Tricks.pdf
│   └── South of France - Traditions and Culture.pdf
├── output1a/                       # Extracted PDF structures (JSON)
│   ├── South of France - Cities.json
│   ├── South of France - Cuisine.json
│   └── ...
├── models--sentence-transformers--all-MiniLM-L6-v2/  # Local ML model
├── main1a.py                       # PDF structure extraction
├── main.py                         # Travel planning analysis
├── test.py                         # Text chunking utilities
├── challenge1b_input.json          # Input specifications
├── challenge1b_output.json         # Final recommendations
├── requirements.txt                # Python dependencies
└── Dockerfile                      # Container configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required Python packages (see requirements.txt)

### Installation

1. **Clone or download the project**

   ```bash
   cd ah-21b
   ```

2. **Install dependencies**

   ```bash
   pip install PyMuPDF pymupdf4llm nltk sentence-transformers scikit-learn torch chromadb matplotlib
   ```

3. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

### Usage

#### Step 1: Extract PDF Structure

Process all travel guide PDFs and extract their structural information:

```bash
python main1a.py
```

This creates JSON files in `output1a/` with extracted titles, headings, and page numbers.

#### Step 2: Generate Travel Recommendations

Analyze the extracted content and generate personalized recommendations:

```bash
python main.py
```

This processes the challenge input and creates `challenge1b_output.json` with travel recommendations.

## 📝 Input/Output Format

### Input (`challenge1b_input.json`)

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner",
    "description": "France Travel"
  },
  "documents": [
    {
      "filename": "South of France - Cities.pdf",
      "title": "South of France - Cities"
    }
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  }
}
```

### Output (`challenge1b_output.json`)

```json
{
    "metadata": {
        "input_documents": ["PDFs/South of France - Cities.pdf", ...],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
        "processing_timestamp": "2025-07-29T00:01:38.185515"
    },
    "extracted_sections": [
        {
            "document": "PDFs/South of France - Tips and Tricks.pdf",
            "section_title": "Conclusion",
            "importance_rank": 1,
            "page_number": 8
        }
    ],
    "subsection_analysis": [
        {
            "document": "PDFs/South of France - Tips and Tricks.pdf",
            "refined_text": "Comprehensive packing guidance...",
            "page_number": 8
        }
    ]
}
```

## 🔧 Configuration

### Model Configuration

The system uses a local sentence transformer model located at:

```
models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/
```

### Text Chunking Parameters

- **Context Window**: 1 sentence on each side
- **Similarity Threshold**: 95th percentile for breakpoint detection
- **Minimum Chunk Size**: 3 sentences
- **Content Filtering**: Minimum 400 characters for analysis

## 🛠️ Technical Components

### main1a.py - PDF Structure Extraction

- Extracts document titles and hierarchical headings
- Handles both markdown-style (#) and bold text (**text**) formatting
- Maps content to accurate page numbers
- Outputs structured JSON with heading levels (H1-H4)

### main.py - Travel Planning Analysis

- Loads extracted PDF structures
- Performs semantic text chunking
- Uses ChromaDB for vector similarity search
- Ranks content by relevance to travel planning query
- Generates top 5 recommendations

### test.py - Text Chunking Utilities

- Semantic sentence segmentation
- Context-aware embedding generation
- Distance-based breakpoint identification
- Small chunk merging for coherent sections

## 📊 Performance

- **Processing Speed**: ~30 seconds for 7 PDFs (structure extraction)
- **Analysis Time**: ~45 seconds for semantic analysis and ranking
- **Model Size**: ~87MB (sentence transformer)
- **Memory Usage**: ~2GB peak during processing

## 🐳 Docker Support

The project is fully containerized with a comprehensive Docker setup that handles all dependencies and provides multiple usage modes.

### Building the Docker Image

```bash
docker build -t travel-planner .
```

### Usage Options

#### 1. Complete Pipeline (Recommended)

Process PDFs and generate travel recommendations in one command:

```bash
# Windows/PowerShell
docker run --rm -v ${PWD}/PDFs:/app/PDFs -v ${PWD}/output1a:/app/output1a -v ${PWD}:/app/host travel-planner

# Linux/Mac
docker run --rm -v $(pwd)/PDFs:/app/PDFs -v $(pwd)/output1a:/app/output1a -v $(pwd):/app/host travel-planner
```

#### 2. PDF Structure Extraction Only

Extract headings and structure from PDFs:

```bash
# Windows/PowerShell
docker run --rm -v ${PWD}/PDFs:/app/PDFs -v ${PWD}/output1a:/app/output1a travel-planner /app/extract_pdfs.sh

# Linux/Mac
docker run --rm -v $(pwd)/PDFs:/app/PDFs -v $(pwd)/output1a:/app/output1a travel-planner /app/extract_pdfs.sh
```

#### 3. Travel Analysis Only

Generate recommendations from pre-extracted data:

```bash
# Windows/PowerShell
docker run --rm -v ${PWD}/output1a:/app/output1a -v ${PWD}:/app/host travel-planner /app/analyze_travel.sh

# Linux/Mac
docker run --rm -v $(pwd)/output1a:/app/output1a -v $(pwd):/app/host travel-planner /app/analyze_travel.sh
```

#### 4. Interactive Development

For debugging and development:

```bash
docker run --rm -it -v ${PWD}:/app/host travel-planner bash
```

### Volume Mounts Explained

- `/app/PDFs` - Input PDF directory
- `/app/output1a` - Extracted PDF structures (JSON files)
- `/app/host` - Access to host filesystem for output files
- Container automatically copies `challenge1b_output.json` to host

### Docker Features

- **Pre-installed Dependencies**: All Python packages and system libraries
- **NLTK Data**: Automatically downloaded punkt tokenizers
- **Local ML Model**: Sentence transformer model included in image
- **Health Checks**: Container health monitoring
- **Multiple Entry Points**: Flexible execution modes
- **Optimized Build**: Multi-stage build with .dockerignore

### Docker Compose (Alternative)

For easier container management, use Docker Compose:

```bash
# Complete pipeline
docker-compose up travel-planner

# PDF extraction only
docker-compose --profile pdf-only up pdf-extractor

# Travel analysis only
docker-compose --profile analysis-only up travel-analyzer

# Build and run in one command
docker-compose up --build

# Clean up
docker-compose down --rmi local
```

### Container Output

The container will output progress information and file locations:

```
Starting Travel Planning Document Processing...
Step 1: Extracting PDF structures...
Step 2: Generating travel recommendations...
Processing complete! Check output files:
- PDF structures: /app/output1a/
- Travel recommendations: /app/challenge1b_output.json
```

## 🤝 Contributing

1. Ensure all dependencies are installed
2. Test with sample PDFs before making changes
3. Maintain compatibility with the input/output JSON formats
4. Update documentation for any new features

## 📄 License

This project is designed for travel planning document analysis and educational purposes.

## 🆘 Troubleshooting

### Common Issues

**Model Loading Errors**

- Ensure the local model directory exists with correct permissions
- Check that all model files are present in the snapshots directory

**PDF Processing Failures**

- Verify PDFs are not corrupted or password protected
- Ensure PDFs contain extractable text (not just images)

**Memory Issues**

- Reduce batch size for large document sets
- Consider processing PDFs individually for very large files

**NLTK Data Missing**

- Run: `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`

### Debug Mode

Enable verbose logging by uncommenting the logging configuration in `main1a.py`:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

## 📞 Support

For technical issues or questions about the travel planning system, please check the approach.md file for detailed technical documentation.
