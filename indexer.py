import logging
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, UnstructuredExcelLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomJSONLoader(JSONLoader):
    def _get_text(self, sample):
        if isinstance(sample, dict):
            if 'content' in sample:
                return sample['content']
            else:
                logger.warning("Sample does not have the expected structure.")
                return json.dumps(sample)
        return super()._get_text(sample)

def load_documents(directory: str) -> list:
    """Load documents from the specified directory."""
    # JSON Loader
    json_loader = DirectoryLoader(
        directory,
        glob="*.json",
        loader_cls=CustomJSONLoader,
        loader_kwargs={'jq_schema': '.'}
    )
    
    # Excel Loader
    excel_loader = DirectoryLoader(
        directory,
        glob="*.xlsx",
        loader_cls=UnstructuredExcelLoader
    )
    
    logger.debug("Loading data from directory...")
    json_data = json_loader.load()
    excel_data = excel_loader.load()
    
    # Combine both document types
    all_documents = json_data + excel_data
    logger.debug(f"Loaded {len(all_documents)} documents total.")
    return all_documents

# Main execution
if __name__ == "__main__":
    # Load documents
    documents = load_documents('./data')

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

    # Create Semantic Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7500,
        chunk_overlap=100
    )

    # Split documents into chunks
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./db-nomic"
    )

    logger.debug("Vectorstore created")