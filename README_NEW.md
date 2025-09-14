# Student Assistant RAG System

A comprehensive AI-powered system that helps students find information about Indian colleges, universities, courses, and admissions using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **Multi-Source Knowledge Retrieval**: Combines ChromaDB, WebSearchAPI, Search1API, NewsData.io and Gemini Pro fallback
- **Comprehensive Information**: College details, courses, admissions, fees, and career guidance
- **Real-time Web Search**: Fetches latest information when needed
- **Smart Relevance Ranking**: Returns most relevant results from all APIs simultaneously
- **Intelligent Fallback**: Graceful degradation when sources are unavailable
- **Student-Friendly Interface**: Clean Streamlit UI optimized for students

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   RAG Pipeline   │    │  Data Sources   │
│                 │    │                  │    │                 │
│ • Chat Interface│────│ 1. Retriever     │────│ • ChromaDB      │
│ • Smart Ranking │    │ 2. Generator     │    │ • WebSearchAPI  │
│ • Error Handling│    │ 3. Prompt Mgmt   │    │ • Search1API    │
│                 │    │                  │    │ • NewsData.io   │
│                 │    │                  │    │ • Gemini Pro    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google Gemini API Key
- Brave Search API Key (optional but recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Student-Assistant-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here
   BRAVE_SEARCH_API_KEY=your_brave_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### API Keys

- **Google Gemini API**: Required for embeddings and text generation
  - Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
  
- **Brave Search API**: Optional but recommended for real-time web search
  - Get your key from [Brave Search API](https://brave.com/search/api/)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_GEMINI_API_KEY` | Yes | Gemini API key for embeddings and generation |
| `BRAVE_SEARCH_API_KEY` | No | Brave Search API for web search |

## 🔄 RAG Pipeline Workflow

1. **Document Retrieval**
   - Search ChromaDB vector database first
   - If insufficient context, query Brave Search API
   - Scrape and process web content
   - Add new content to ChromaDB for future use

2. **Fallback Chain**
   - ChromaDB → WebSearchAPI → Search1API → NewsData.io → Gemini Pro Internal Knowledge
   - Each step has error handling and graceful degradation

3. **Response Generation**
   - Use retrieved context with Gemini model
   - Format response with proper structure
   - Include source attribution with relevance scores
   - Smart ranking of results from all sources

## 📁 Project Structure

```
Student-Assistant-System/
├── app.py                      # Main Streamlit application
├── google_gemini.py           # Gemini API integration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── rag_pipeline/              # RAG system modules
│   ├── __init__.py
│   ├── retriever.py          # Document retrieval logic
│   ├── generator.py          # Response generation
│   └── prompt_template.py    # Prompt templates
├── db-minlm-v2/              # ChromaDB storage
└── data/                     # College data (JSON files)
```

## 🎯 Usage Examples

### Basic Queries
- "Tell me about KL University"
- "IIT admission requirements"
- "Engineering colleges in Bangalore"
- "MBA programs and fees"

### Detailed Queries
- "What are the eligibility criteria for Computer Science at VIT?"
- "Compare fees between private and government engineering colleges"
- "Placement statistics for mechanical engineering courses"

## 🔍 Key Components

### DocumentRetriever (`rag_pipeline/retriever.py`)
- ChromaDB integration with similarity search
- WebSearchAPI.ai, Search1API, NewsData.io integration
- Smart relevance ranking and deduplication
- Gemini Pro fallback knowledge

### ResponseGenerator (`rag_pipeline/generator.py`)
- Response formatting and cleaning
- Source attribution with relevance scores
- Error handling and fallback responses

### PromptTemplates (`rag_pipeline/prompt_template.py`)
- System prompts for student assistance
- Context-aware user prompts
- Specialized templates for different query types

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are correctly set in `.env`
   - Verify API key permissions and quotas

2. **ChromaDB Initialization**
   - Delete `db-minlm-v2` folder and restart if corrupted
   - Check disk space for database storage

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version compatibility

4. **Slow Responses**
   - Check internet connection for web search
   - Consider reducing search result count
   - Monitor API rate limits

### Performance Optimization

- **ChromaDB**: Regular database cleanup and indexing
- **Web Search**: Implement caching for frequently asked queries
- **API Calls**: Batch requests when possible
- **Response Time**: Use async operations for concurrent processing

## 📈 Future Enhancements

- [ ] Integration with more educational APIs
- [ ] Voice input/output capabilities
- [ ] Personalized recommendations based on user preferences
- [ ] Integration with college admission portals
- [ ] Analytics and usage tracking
- [ ] Advanced filtering and search options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for AI capabilities
- Brave Search for real-time web data
- ChromaDB for vector storage
- Streamlit for the web interface
- CollegeDunia for educational data

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Note**: This system provides educational information and should not be the sole source for important decisions. Always verify information from official university sources.
