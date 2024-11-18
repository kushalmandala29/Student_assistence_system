import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from dotenv import load_dotenv
import aiohttp
import brotli
import chromadb
import nltk
import streamlit as st
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Request rate and concurrency
API_MAX_CONCURRENT_REQUESTS = 5 
API_RPS = 1 # Rate limit of 1 request per second
API_RATE_LIMIT = AsyncLimiter(API_RPS, time_period=1)
API_TIMEOUT = 60  # Increase timeout to 60 seconds

# Brave Search Key
API_KEY =  os.getenv("API_KEY")

# Brave Search API host and paths
API_HOST = "https://api.search.brave.com"
API_PATH = {
    "web": urljoin(API_HOST, "res/v1/web/search"),
}

# Create request headers
API_HEADERS = {
    "web": {
        "X-Subscription-Token": API_KEY,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/json",
    },
}

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the vector database with similarity search
def initialize_vector_db():
    """
    Initialize and configure a vector database using ChromaDB.
    
    This function:
    1. Creates a persistent database directory
    2. Initializes a ChromaDB client with specific settings
    3. Sets up a Chroma collection with embedding functionality
    4. Verifies the collection exists and contains documents
    5. Initializes with a dummy document if empty
    
    Returns:
        Chroma: Initialized vector database instance
        
    Raises:
        RuntimeError: If database initialization fails
    """
    try:
        # Create db directory if it doesn't exist
        db_dir = Path("./db-minlm-v2")  # Changed from db-nomic to db-minlm-v2
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        # Initialize Chroma with explicit client
        db = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
        
        # Verify collection exists and is accessible
        try:
            count = db._collection.count()
            logger.info(f"Found {count} documents in vector store")
            
            # Initialize with dummy document if empty
            if count == 0:
                logger.info("Creating empty collection with initialization document")
                db.add_documents([Document(page_content="Initialization document")])
                
            return db
            
        except Exception as collection_error:
            logger.error(f"Collection error: {str(collection_error)}")
            # Try to reset collection if there's an error
            client.reset()
            db = Chroma(
                client=client,
                collection_name="langchain",
                embedding_function=embeddings,
            )
            db.add_documents([Document(page_content="Initialization document")])
            return db
            
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        raise RuntimeError(f"Vector database initialization failed: {str(e)}")

# Initialize the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    show_progress=True
)

# Initialize the Llama model
local_llm = 'llama3.2'
llm = ChatOllama(
    model=local_llm,
    keep_alive="3h",
    max_tokens=512,
    temperature=0
)

template = """<bos><start_of_turn>user\nYou are a Student Assistant System designed to help students with information regarding colleges, universities, ug and pg programmes with branch ,courses including each specializations, and their careers. \
Please provide a meaningful response based on the following context. Ensure your answer is in full sentences with correct spelling and punctuation. \
Format your response with bullet points and sections where appropriate. \
Do not include phrases like 'Based on the provided context' or 'I found information about'. Instead, start directly with the relevant information. \
If the context does not contain the answer, respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)


def normalize_score(score):
    """Convert cosine distance to similarity score between 0 and 1
    
    Args:
        score (float): The cosine distance score to normalize
        
    Returns:
        float: Normalized similarity score between 0 and 1
    """
    # Cosine distance to similarity conversion
    return 1 - (score / 2)

def check_internal_knowledge(prompt):
    """
    Check if the given prompt has relevant matches in the vector database.
    
    This function performs a similarity search in the vector database and evaluates
    the relevance of the results based on normalized similarity scores.
    
    Args:
        prompt (str): The input text to search for in the vector database
        
    Returns:
        tuple: A tuple containing:
            - bool: True if relevant documents are found above threshold, False otherwise
            - list: List of relevant documents if found, empty list otherwise
            
    The function performs the following steps:
    1. Executes similarity search with scores against the vector database
    2. Normalizes the similarity scores to a 0-1 range
    3. Calculates average normalized score
    4. Returns results if average score meets minimum threshold (0.6)
    """
    try:
        # Use standard similarity search with scores
        results = db.similarity_search_with_score(
            prompt, 
            k=5
        )
        if not results:
            logger.info("No relevant documents found in vector store.")
            return False, []
        
        # Process and normalize scores
        docs_and_scores = []
        max_score = max(score for _, score in results) if results else 1
        min_score = min(score for _, score in results) if results else 0
        
        # Normalize scores to [0,1] range
        for doc, score in results:
            if max_score == min_score:
                normalized_score = 1.0 if score == max_score else 0.0
            else:
                normalized_score = 1 - ((score - min_score) / (max_score - min_score))
            docs_and_scores.append((doc, normalized_score))
        
        # Calculate average normalized score
        scores = [score for _, score in docs_and_scores]
        avg_score = sum(scores) / len(scores) if scores else 0
        logger.info(f"Average normalized similarity score: {avg_score}")
        
        # Extract documents
        docs = [doc for doc, _ in docs_and_scores]
        
        threshold = 0.6
        return avg_score >= threshold, docs
    except Exception as e:
        logger.error(f"Error checking vector database: {str(e)}")
        return False, []

# Initialize vector database
db = initialize_vector_db()
if not db:
    logger.error("Failed to initialize vector database")
    raise RuntimeError("Vector database initialization failed")

# Update retriever configuration
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print(f"Number of documents in vector store: {db._collection.count()}")

def clean_text(text):
    """
    Clean and preprocess text by removing stop words and lemmatizing tokens.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def chunk_text(text, chunk_size=1500, chunk_overlap=300):
    """
    Split input text into smaller overlapping chunks for processing.

    Args:
        text (str): The input text to be split into chunks
        chunk_size (int, optional): Maximum size of each text chunk in characters. Defaults to 1500.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 300.

    Returns:
        list: A list of text chunks with specified size and overlap
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_text(text)
    return chunks


async def ensure_ollama_connection(max_retries=3, delay=2):
    """Ensure Ollama service is running with retries"""
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    if response.status == 200:
                        return True
        except Exception:
            if attempt < max_retries - 1:
                logger.warning(f"Ollama connection attempt {attempt + 1} failed, retrying in {delay}s...")
                await asyncio.sleep(delay)
            continue
    return False


def add_to_vector_store(chunks):
    """
    Add new text chunks as documents to the existing vector store database.
    
    Args:
        chunks (list): List of text chunks to be added as documents
        
    Returns:
        bool: True if documents were successfully added, False otherwise
        
    The function:
    1. Converts text chunks to Document objects
    2. Adds the documents to the existing vector store
    3. Logs the number of documents added and total count
    4. Handles errors during document addition
    """
    try:
        # Ensure chunks are properly converted to strings
        documents = [
            Document(
                page_content=str(chunk) if not isinstance(chunk, str) else chunk,
                metadata={"source": chunk.metadata["source"] if hasattr(chunk, "metadata") else None}
            ) 
            for chunk in chunks
        ]
        
        if not documents:
            logger.info("No documents to add to the vector store.")
            return False

        db.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} documents to vector store")
        return True
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
        return False

async def scrape_data(url, session, retries=0):
    """
    Asynchronously scrapes text content from a given URL using an aiohttp session.
    
    Args:
        url (str): The URL to scrape content from
        session (aiohttp.ClientSession): An active aiohttp session for making requests
        retries (int, optional): Number of retry attempts if scraping fails. Defaults to 0
        
    Returns:
        str: The scraped text content from all paragraph elements, or empty string if scraping fails
        
    The function:
    1. Sets up request headers to mimic a browser
    2. Makes an async GET request to the URL
    3. Parses the HTML content using BeautifulSoup
    4. Extracts text from all paragraph elements
    5. Handles various errors like timeouts and connection issues
    """
    headers = {
        'referer': 'https://www.scrapingcourse.com/ecommerce/',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'sec-ch-device-memory': '8',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-platform': 'Windows',
        'sec-ch-ua-platform-version': '"10.0.0"',
        'sec-ch-viewport-width': '792',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    }
    try:
        async with API_RATE_LIMIT:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve data from {url}. Status code: {response.status}")
                    return ''
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                paragraphs = soup.find_all('p')
                if not paragraphs:
                    logger.info(f"No content found at {url}.")
                    return ''
                text_content = ' '.join([p.get_text() for p in paragraphs])
                return text_content
    except asyncio.TimeoutError:
        logger.warning(f"Timeout error for {url}. Skipping...")
        return ''
    except aiohttp.client_exceptions.ClientOSError as e:
        logger.error(f"Connection error: {str(e)}. Skipping...")
        return ''
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}. Skipping...")
        return ''
    return ''

async def update_vector_store_with_scraped_data(url, session):
    """
    Update vector store with new scraped data from a given URL.
    
    This function:
    1. Scrapes text content from the provided URL
    2. Cleans and chunks the text content
    3. Creates Document objects with source metadata
    4. Adds the documents to the vector store
    
    Args:
        url (str): The URL to scrape data from
        session (aiohttp.ClientSession): An active aiohttp session for making requests
        
    Returns:
        bool: True if the update was successful, False otherwise
        
    Raises:
        Exception: Any unexpected errors during the process are caught and logged
    """
    try:
        text_content = await scrape_data(url, session)
        if not text_content:
            logger.info(f"No text content retrieved from {url}. Skipping.")
            return False

        cleaned_text = clean_text(text_content)
        chunks = chunk_text(cleaned_text)
        if not chunks:
            logger.info(f"No chunks generated from {url}. Skipping.")
            return False

        # Add source metadata to documents
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": url}
            ) for chunk in chunks
        ]
        
        success = add_to_vector_store(documents)
        if success:
            logger.info(f"Successfully processed and stored data from {url}")
        return success
    except Exception as e:
        logger.error(f"Error updating vector store with scraped data: {str(e)}")
        return False

async def extract_college_name(query):
    """
    Extracts the college/university name from a given query using ChatOllama LLM.
    
    This function:
    1. Uses ChatOllama to intelligently extract college names from user queries
    2. Falls back to simple string parsing if LLM extraction fails
    
    Args:
        query (str): The input query containing a college/university name
        
    Returns:
        str: The extracted college name
        
    Raises:
        Exception: Any errors during LLM processing are caught and logged,
                  falling back to simple extraction
    """
    # Use ChatOllama to extract the college name from the query
    system_message = "You are a helpful assistant that extracts college/university names from queries. Only return the college name, nothing else."
    user_message = f"Extract only the college name from this query: {query}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = await llm.ainvoke(messages)
        college_name = response.content.strip()
        return college_name
    except Exception as e:
        logger.error(f"Error extracting college name: {str(e)}")
        # Fallback to simple extraction if LLM fails
        if "by" in query.lower():
            return query.split("by")[-1].strip()
        return query.strip()

def clean_college_name(college_name: str) -> str:
    """
    Clean and normalize college name for URL construction.
    
    Args:
        college_name (str): Raw college name
        
    Returns:
        str: Cleaned college name in format 'nameuniversity'
        
    Examples:
        "K L University" -> "kluniversity"
        "KL University" -> "kluniversity"
        "K.L. University" -> "kluniversity"
    """
    # Remove special characters and dots
    college_name = re.sub(r'[^a-zA-Z0-9\s]', '', college_name)
    
    # Remove common words except university
    college_name = re.sub(r'\b(college|institute)\b', '', 
                         college_name, flags=re.IGNORECASE)
    
    # Convert to lowercase and remove spaces
    college_name = ''.join(college_name.lower().split())
    
    # Ensure consistent "university" suffix
    if "university" not in college_name:
        college_name += "university"
    
    return college_name

async def generate_dork_query(query):
    """
    Generate a Google dork query to find educational content related to a college/university.
    
    Args:
        query (str): User input query containing college/university name
        
    Returns:
        str: Formatted Google dork query targeting educational domains and content
        
    The function:
    1. Extracts college name using LLM
    2. Cleans/normalizes the college name
    3. Constructs search query targeting:
        - Official .edu/.ac.in/.in domains
        - Pages containing college name
        - Educational content keywords
    """
    try:
        cn = await extract_college_name(query)
        cnc = clean_college_name(cn)
        
        # Simplified and more focused dork query
        dork_query = (
            f'(site:"{cnc}.edu" OR '
            f'site:"{cnc}.ac.in" OR '
            f'site:"{cnc}.in") '
            f'(intext:"{cn}" '
            'intext:"courses offered" OR '
            'intext:"admission" OR '
            'intext:"programs" OR '
            'intext:"btech" OR '
            'intext:"engineering" OR '
            'intext:"Fee Structure")'
        ).replace('\n', '').strip()
        
        logger.debug(f"Generated dork query: {dork_query}")
        return dork_query
    except Exception as e:
        logger.error(f"Error generating dork query: {str(e)}")
        # Fallback to basic search
        return f'site:"{clean_college_name(query)}.in" OR site:"{clean_college_name(query)}.ac.in"'

async def search(query, session):
    """
    Performs an asynchronous web search using the provided query and session.
    
    Args:
        query (str): The search query string to be executed
        session (aiohttp.ClientSession): The aiohttp session for making HTTP requests
        
    Returns:
        dict: A dictionary containing search results with the following structure:
            {
                "type": "search",
                "results": [],
                "urls": [list of up to 5 valid URLs]
            }
            
    The function:
    1. Makes an API request with rate limiting
    2. Processes the JSON response
    3. Filters out document file types (.pdf, .doc, etc.)
    4. Limits results to 5 valid URLs
    5. Handles various error cases (rate limits, connection errors)
    """
    params = {
        "q": query,
        "summarize": "true",
        "format": "json",
        "summary": "true",
        "count": 5  # Limit to 5 results
    }
    try:
        async with API_RATE_LIMIT:
            async with session.get(API_PATH["web"], params=params, headers=API_HEADERS["web"]) as response:
                logger.info(f"Querying URL: {response.url}")
                if response.status == 200:
                    data = await response.json()
                    formatted_results = {
                        "type": "search",
                        "results": [],
                        "urls": []
                    }
                    results = data.get("web", {}).get("results", [])
                    # Limit to first 5 valid URLs
                    url_count = 0
                    for result in results:
                        url = result.get("url", "")
                        if url and url_count < 5 and not any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls']):
                            formatted_results["urls"].append(url)
                            url_count += 1
                    print(formatted_results)
                    return formatted_results
                elif response.status == 429:
                    logger.error("Rate Limit Exceeded")
                    return {"type": "search", "results": [], "urls": []}
                else:
                    logger.error(f"Failed to retrieve data. Status code: {response.status}")
                    return {"type": "search", "results": [], "urls": []}
    except aiohttp.client_exceptions.ClientOSError as e:
        logger.error(f"Connection error: {str(e)}")
        return {"type": "search", "results": [], "urls": []}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

        return {"type": "search", "results": [], "urls": []}

async def is_valid_prompt(prompt):
    """
    Asynchronously validates if a given prompt is appropriate and safe to use.
    
    This function uses an AI model to check if the input prompt contains any explicit
    language or out-of-context content that should be filtered out.
    
    Args:
        prompt (str): The user input prompt to validate
        
    Returns:
        int: Returns 1 if the prompt is valid and safe to use, 0 if it contains
             inappropriate content. Returns 1 by default if validation fails.
             
    Raises:
        No exceptions are raised as they are caught and logged internally
    """
    # Check for explicit language or out-of-context prompts
    system_message = "You are a helpful assistant that checks if a query is valid and no explicit words or language are used. Return 1 if the query is valid and 0 if it is not. Only return a number."
    user_message = f"Is this query valid? {prompt}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = await llm.ainvoke(messages)
        validity = response.content.strip()
        return int(validity)
    except Exception as e:
        logger.error(f"Error checking prompt validity: {str(e)}")
        return 1
def clearChatHistory():
        """
        Clears the chat history in the Streamlit session state and resets it to the initial greeting message.
    
        This function removes all existing messages from the chat history and sets it back to only
        contain the default assistant greeting message.
    
        Returns:
            None
        """
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you out today?"}]

def generateResponse(promptInput, context_docs=None):
    """
    Generates an AI response based on the provided prompt and optional context documents.
    
    This function processes the input prompt along with any provided context documents,
    generates a response using an LLM, cleans and formats the output, and adds citations
    if sources are available.
    
    Args:
        promptInput (str): The user's input prompt/question to be answered
        context_docs (list, optional): List of document objects containing relevant context
            and metadata. Each document should have page_content and optionally metadata
            with a 'source' field. Defaults to None.
            
    Returns:
        str: The formatted response string, including citations if sources are available.
            Returns an error message if processing fails.
            
    Raises:
        No exceptions are raised as they are caught and logged internally.
    """
    logger.debug(f"[START] generateResponse for: {promptInput}")
    try:
        if context_docs is None:
            context = "No context available."
            sources = []
        else:
            # Extract sources from metadata if available
            sources = []
            context_parts = []
            for doc in context_docs:
                context_parts.append(doc.page_content)
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
            context = "\n\n".join(context_parts)
            
        logger.debug(f"[TEMPLATE] Using context with {len(context)} chars")
        prompt_text = template.format(context=context, question=promptInput)
        
        messages = [{"role": "user", "content": prompt_text}]
        response = llm.invoke(messages)
        output = response.content if hasattr(response, 'content') else str(response)
        
        # Clean and format the response
        cleaned_output = (output.replace("<bos>", "")
                              .replace("<start_of_turn>user", "")
                              .replace("<start_of_turn>model", "")
                              .replace("<end_of_turn>", "")
                              .replace("ANSWER:", "")
                              .replace("Based on the provided context,", "")
                              .replace("Based on the information provided,", "")
                              .replace("I found information about", "")
                              .strip())
        
        # Format and add citations
        if cleaned_output and sources:
            formatted_lines = []
            lines = cleaned_output.split('\n')
            
            # Format main content
            for line in lines:
                if line.strip() and not any(line.strip().startswith(x) for x in ['•', '-', '*', '1.', '2.', '3.', '4.', '5.']):
                    line = f"• {line.strip()}"
                formatted_lines.append(line)
            
            # Add sources section
            if sources:
                formatted_lines.append("\n**Sources:**")
                for i, source in enumerate(set(sources), 1):
                    formatted_lines.append(f"[{i}] {source}")
            
            cleaned_output = '\n'.join(formatted_lines)
            
        logger.debug(f"[FINAL] Formatted response with citations: {cleaned_output}")
        return cleaned_output if cleaned_output else "I apologize, I couldn't find relevant information to answer your question."
        
    except Exception as e:
        logger.error(f"[ERROR] Response generation failed: {str(e)}")
        return "I encountered an error while processing your request. Please try again."

def display_streamed_response(response, placeholder):
    """
    Helper function to stream and format response text with visual formatting.
    Args:
        response (str): The text response to be streamed and formatted
        placeholder (streamlit.empty): A Streamlit empty placeholder element to display the text
        
    Returns:
        str: The final formatted display text
    
    The function:
        - Takes each character from the response and builds it gradually
        - Applies formatting like numbered headers (1-5) and bullet points
        - Adds visual emphasis to "Please note" text
        - Shows a cursor (▌) while streaming
        - Uses a small delay between characters for a typing effect
        - Updates the placeholder with formatted markdown on each iteration
    """
    formatted_response = ""
    for char in response:
        formatted_response += char
        display_text = formatted_response.replace("1.", "\n### 1.") \
                                         .replace("2.", "\n### 2.") \
                                         .replace("3.", "\n### 3.") \
                                         .replace("4.", "\n### 4.") \
                                         .replace("5.", "\n### 5.") \
                                         .replace("- ", "\n- ") \
                                         .replace("Please note", "\n\n*Please note*")
        placeholder.markdown(display_text + "▌")
        time.sleep(0.01)
    placeholder.markdown(display_text)
    return display_text

async def handle_information_query(prompt):
    """
    Handle user queries by checking internal knowledge and performing web searches if needed.
    
    Args:
        prompt (str): The user's input query text
        
    Returns:
        str: The formatted response text
        
    The function:
        1. Validates the prompt for appropriate content
        2. Checks internal knowledge base first
        3. If no internal knowledge, performs web search
        4. Processes and stores new information from search results
        5. Generates and streams formatted response to user
    """
    
    # Check for explicit language
    if not await is_valid_prompt(prompt):
        st.warning("Sorry, I can't assist with that.")
        return "I apologize, I cannot assist with that request."

    # First check internal knowledge
    has_knowledge, docs = check_internal_knowledge(prompt)
    
    placeholder = st.empty()
    display_text = ""  # Ensure display_text is always defined
    
    if has_knowledge:
        logger.info("Using internal knowledge for response")
        with st.spinner("🤖 Generating response from existing knowledge..."):
            response = generateResponse(prompt, docs)
            display_text = display_streamed_response(response, placeholder)
            return display_text

    # If no internal knowledge, perform web search
    logger.info("No internal knowledge found, searching web...")
    with st.spinner("🌐 Searching external sources..."):
        async with ClientSession(connector=TCPConnector(limit=API_MAX_CONCURRENT_REQUESTS),
                               timeout=ClientTimeout(total=API_TIMEOUT)) as session:
            dork_query = await generate_dork_query(prompt)
            search_results = await search(dork_query, session)
            
            if not search_results or not search_results.get("urls"):
                logger.warning("No search results found")
                response = generateResponse(prompt)
                display_text = display_streamed_response(response, placeholder)
                return display_text

            # Process search results
            with st.spinner("📚 Processing new information..."):
                tasks = []
                for url in search_results.get("urls", []):
                    task = asyncio.create_task(update_vector_store_with_scraped_data(url, session))
                    tasks.append(task)
                
                update_results = await asyncio.gather(*tasks)
                
                # Check if any updates were successful
                if any(update_results):
                    # Re-check internal knowledge after updates
                    has_knowledge, docs = check_internal_knowledge(prompt)
                    response = generateResponse(prompt, docs if has_knowledge else None)
                else:
                    logger.warning("No new data could be processed")
                    response = generateResponse(prompt)

                display_text = display_streamed_response(response, placeholder)
                return display_text
            
async def main():
    """
    Main function to run the Student Assistance System Streamlit app.

    The application serves as a smart assistant to help students pursue their studies in dream colleges.

    Features:
        - Sidebar for navigation and information
        - Chat interface for student queries
        - Chat history storage and clearing functionality
        - Smart responses to greetings or specific queries using AI

    Functionality:
        - Configures the page title for Streamlit
        - Handles sidebar content, including title, description, and a button to clear chat history
        - Stores and displays chat history using session state
        - Captures user prompts and generates responses (static or dynamic) based on input
        - Displays AI responses in the chat interface
    """
    # Set the page configuration with a title
    st.set_page_config(page_title="Student Assistance System")

    # Sidebar setup
    with st.sidebar:
        # Sidebar title and description
        st.title("Student Assistant System")
        st.write("By KL GLUG")
        st.write("A smart assistance to help students pursue their studies in dream colleges.")
        
        # Navigation link to chat and a button to clear chat history
        st.markdown("[Chat Now!](#chat)")
        st.button('Clear Chat History', on_click=clearChatHistory)

    # Store chat history in session state
    if "messages" not in st.session_state:
        # Initialize session state with a default assistant message
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you out today?"}]

    # Display chat history or clear messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display each message from the chat history
            st.write(message["content"])

    # Capture user input in the chat interface
    if prompt := st.chat_input():
        # Append the user's prompt to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display a response from the assistant
        with st.chat_message("assistant"):
            # Check if the user's input is a greeting and respond appropriately
            if any(greeting in prompt.lower() for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
                response = "Hello! How can I assist you today?"
                st.write(response)
            else:
                # Handle other queries using an asynchronous function
                response = await handle_information_query(prompt)

        # Append the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    asyncio.run(main())