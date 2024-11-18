# Student Assistant System

An AI-powered RAG (Retrieval-Augmented Generation) system that helps students find information about colleges, courses, and educational programs.

**Note**: The data used in this project is sourced from CollegeDunia.com and is used solely for educational purposes. We do not intend to use it for any commercial or other purposes.

## 🚀 Features

- Real-time college information retrieval
- Course and program details lookup
- Fee structure information
- Dynamic web scraping
- Persistent knowledge base
- Conversational UI

## 📋 Prerequisites

- Python 3.9+
- Ollama (for LLM and Embeddings)
- ChromaDB (for vector storage)
- Internet connection

## 🛠️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Create virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    ```bash
    cp .env.example .env
    ```
    ```
    API_KEY=your-brave-search-key
    ```
    Soure: https://brave.com/search/api/
    

4. Download and Install Ollama:
    ```bash
    https://ollama.com/
    ```

5. Setup LLM Model and Embedding Model:
    ```bash
    ollama pull llama3.1
    ollama pull nomic-embed-text
    ```

6. Deploy Ollama as API:
    ```bash
    ollama serve
    ```

7. Run the application:
    ```bash
    streamlit run ui.py
    ```

## 📝 Usage

1. Open the application in your web browser.
2. Enter a question related to colleges, courses, or educational programs.
3. The system will retrieve relevant information from the knowledge base and provide answers.

**Note**: 
This system can give inaccurate or incomplete information. It is intended for educational purposes and should not be used for professional or legal purposes. Always check for the official website for the best information.


## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📝 License
    GNU General Public License v3.0

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/)
- [ChromaDB](https://chromadb.org/)
- [CollegeDunia](https://collegedunia.com/)

## Contributers:
- [Dr.B. Arune Kumar]() - Incharge and Instructor
- [P.M.SALMAN KHAN](https://github.com/PrashantChoudhary)
- [Venkata Kowsic]()
- [M. Kushal]()
- [K Rutvik](https://github.com/PrashantChoudhary)
- [P. Azam Khan](https://github.com/PrashantChoudhary)
- [K. Bathul Sultana](https://github.com/PrashantChoudhary)