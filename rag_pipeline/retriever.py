"""
Retriever module for Student Assistant RAG system.
Handles queries to ChromaDB, WebSearchAPI.ai, Search1API, NewsData.io, and Gemini Pro fallback.
"""

import os
import logging
import asyncio
import aiohttp
import json
import ssl
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google_gemini import GeminiEmbeddings, ChatGoogleGemini
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles document retrieval from multiple sources with fallback chain."""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        self.websearch_api_key = os.getenv("WEBSEARCH_API_KEY")
        self.search1_api_key = os.getenv("SEARCH1_API_KEY")
        self.newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
        
        # Initialize embeddings and vector database
        self.embeddings = GeminiEmbeddings(
            api_key=self.gemini_api_key,
            model="models/embedding-001"
        )
        
        # Initialize ChromaDB
        self.db = self._initialize_vector_db()
        
        # Initialize Gemini for fallback
        self.gemini_fallback = ChatGoogleGemini(
            api_key=self.gemini_api_key,
            model="gemini-1.5-pro"
        )
        
        # API endpoints
        self.websearchapi_url = "https://api.websearchapi.ai/ai-search"
        self.search1api_url = "https://api.search1api.com/search"
        self.newsdata_url = "https://newsdata.io/api/1/latest"
        
        # Rate limiting
        self.rate_limiter = AsyncLimiter(1, time_period=1)  # 1 request per second
    
    def _initialize_vector_db(self) -> Chroma:
        """Initialize ChromaDB vector database."""
        try:
            db_dir = "./db-minlm-v2"
            os.makedirs(db_dir, exist_ok=True)
            
            client = chromadb.PersistentClient(
                path=db_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            db = Chroma(
                client=client,
                collection_name="langchain",
                embedding_function=self.embeddings,
            )
            
            # Initialize with dummy document if empty
            try:
                count = db._collection.count()
                if count == 0:
                    db.add_documents([Document(page_content="Initialization document")])
                logger.info(f"ChromaDB initialized with {count} documents")
            except Exception as e:
                logger.warning(f"ChromaDB collection error: {e}")
                client.reset()
                db = Chroma(
                    client=client,
                    collection_name="langchain",
                    embedding_function=self.embeddings,
                )
                db.add_documents([Document(page_content="Initialization document")])
            
            return db
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"Vector database initialization failed: {e}")
    
    async def retrieve_context(self, query: str, threshold: float = 0.6) -> Tuple[str, List[str]]:
        """
        Main retrieval method that searches all APIs simultaneously and returns the most relevant results.
        Returns (context, sources) tuple with ranked results from all sources.
        """
        sources = []
        all_results = []
        
        # Step 1: Search ChromaDB
        try:
            chroma_context, chroma_sources = await self._search_chromadb(query, threshold)
            if chroma_context:
                all_results.append({
                    'content': chroma_context,
                    'sources': chroma_sources,
                    'api': 'ChromaDB',
                    'score': 1.0  # ChromaDB gets highest priority for exact matches
                })
                logger.info("Successfully retrieved context from ChromaDB")
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
        
        # Step 2: Search all web APIs simultaneously
        api_tasks = []
        
        # WebSearchAPI task
        async def search_websearch():
            try:
                snippets = await self.query_websearchapi(query)
                if snippets:
                    return {
                        'content': snippets,
                        'api': 'WebSearchAPI',
                        'score': 0.9
                    }
            except Exception as e:
                logger.warning(f"WebSearchAPI search failed: {e}")
            return None
        
        # Search1API task
        async def search_search1():
            try:
                snippets = await self.query_search1api(query)
                if snippets:
                    return {
                        'content': snippets,
                        'api': 'Search1API',
                        'score': 0.8
                    }
            except Exception as e:
                logger.warning(f"Search1API search failed: {e}")
            return None
        
        # NewsData.io task
        async def search_newsdata():
            try:
                snippets = await self.query_newsdata(query)
                if snippets:
                    return {
                        'content': snippets,
                        'api': 'NewsData.io',
                        'score': 0.7
                    }
            except Exception as e:
                logger.warning(f"NewsData.io search failed: {e}")
            return None
        
        # Execute all API searches concurrently
        api_tasks = [search_websearch(), search_search1(), search_newsdata()]
        api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
        
        # Process API results
        for result in api_results:
            if result and not isinstance(result, Exception):
                # Calculate relevance score for each snippet
                scored_snippets = []
                for snippet in result['content']:
                    relevance = await self._calculate_relevance(query, snippet)
                    scored_snippets.append({
                        'snippet': snippet,
                        'relevance': relevance,
                        'api': result['api']
                    })
                
                if scored_snippets:
                    all_results.append({
                        'content': scored_snippets,
                        'api': result['api'],
                        'score': result['score']
                    })
                    logger.info(f"Successfully retrieved {len(scored_snippets)} results from {result['api']}")
        
        # Step 3: Rank and combine all results
        final_snippets = []
        final_sources = []
        
        # First add ChromaDB results (highest priority)
        for result in all_results:
            if result['api'] == 'ChromaDB':
                final_snippets.append(result['content'])
                final_sources.extend(result['sources'])
        
        # Then add top-ranked snippets from web APIs
        web_snippets = []
        seen_content = set()  # Track content to avoid duplicates
        
        for result in all_results:
            if result['api'] != 'ChromaDB':
                for item in result['content']:
                    # Create a simplified version for duplicate detection
                    simplified_content = ' '.join(item['snippet'].lower().split()[:10])
                    
                    # Skip if we've seen similar content
                    if simplified_content not in seen_content:
                        seen_content.add(simplified_content)
                        web_snippets.append({
                            'snippet': item['snippet'],
                            'relevance': item['relevance'],
                            'api': item['api'],
                            'api_score': result['score']
                        })
        
        # Sort by combined relevance score (relevance * api_score)
        web_snippets.sort(key=lambda x: x['relevance'] * x['api_score'], reverse=True)
        
        # Take top 6 most relevant snippets from web APIs
        for item in web_snippets[:6]:
            final_snippets.append(item['snippet'])
            final_sources.append(f"{item['api']} - Relevance: {item['relevance']:.2f}")
        
        # Step 4: Gemini Pro fallback if no results
        if not final_snippets:
            try:
                gemini_context = await self._gemini_fallback_knowledge(query)
                if gemini_context:
                    final_snippets.append(gemini_context)
                    final_sources.append("Gemini Pro")
                    logger.info("Using Gemini Pro fallback knowledge")
            except Exception as e:
                logger.error(f"Gemini Pro fallback failed: {e}")
        
        final_context = "\n\n".join(final_snippets) if final_snippets else ""
        return final_context, final_sources
    
    async def _search_chromadb(self, query: str, threshold: float) -> Tuple[str, List[str]]:
        """Search ChromaDB for relevant documents."""
        try:
            results = self.db.similarity_search_with_score(query, k=5)
            if not results:
                return "", []
            
            # Filter by threshold and normalize scores
            relevant_docs = []
            sources = []
            
            max_score = max(score for _, score in results) if results else 1
            min_score = min(score for _, score in results) if results else 0
            
            for doc, score in results:
                if max_score == min_score:
                    normalized_score = 1.0 if score == max_score else 0.0
                else:
                    normalized_score = 1 - ((score - min_score) / (max_score - min_score))
                
                if normalized_score >= threshold:
                    relevant_docs.append(doc.page_content)
                    source = doc.metadata.get("source", "ChromaDB") if hasattr(doc, "metadata") else "ChromaDB"
                    sources.append(source)
            
            context = "\n\n".join(relevant_docs) if relevant_docs else ""
            return context, sources
            
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return "", []
    
    async def query_websearchapi(self, question: str) -> List[str]:
        """
        Query WebSearchAPI.ai and return formatted snippets.
        Returns list of formatted strings (title + snippet).
        """
        if not self.websearch_api_key:
            logger.warning("WebSearchAPI key not configured")
            return []
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.websearch_api_key}"
        }
        
        payload = {
            "query": question,
            "maxResults": 5,
            "includeContent": False,
            "country": "us",
            "language": "en"
        }
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.websearchapi_url,
                        json=payload,
                        headers=headers,
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            # WebSearchAPI.ai returns results in 'organic' field
                            results = data.get("organic", [])
                            
                            snippets = []
                            for result in results:
                                title = result.get("title", "")
                                description = result.get("description", "")
                                if title and description:
                                    formatted_snippet = f"{title}: {description}"
                                    snippets.append(formatted_snippet)
                            
                            return snippets
                        else:
                            logger.warning(f"WebSearchAPI returned status {response.status}")
                            return []
        except Exception as e:
            logger.error(f"WebSearchAPI request failed: {e}")
            return []
    
    async def query_search1api(self, question: str) -> List[str]:
        """
        Query Search1API and return formatted snippets.
        Returns list of formatted strings (title + snippet).
        """
        if not self.search1_api_key:
            logger.warning("Search1API key not configured")
            return []
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.search1_api_key}"
        }
        
        payload = {
            "query": question,
            "search_service": "google",
            "max_results": 5,
            "crawl_results": 0,
            "image": False,
            "include_sites": [],
            "exclude_sites": [],
            "language": "auto"
        }
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.search1api_url,
                        json=payload,
                        headers=headers,
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("results", [])
                            
                            snippets = []
                            for result in results:
                                title = result.get("title", "")
                                snippet = result.get("snippet", "")
                                if title and snippet:
                                    formatted_snippet = f"{title}: {snippet}"
                                    snippets.append(formatted_snippet)
                            
                            return snippets
                        else:
                            logger.warning(f"Search1API returned status {response.status}")
                            return []
        except Exception as e:
            logger.error(f"Search1API request failed: {e}")
            return []
    
    async def query_newsdata(self, question: str) -> List[str]:
        """
        Query NewsData.io and return formatted snippets.
        Returns list of formatted strings (title + description).
        """
        if not self.newsdata_api_key:
            logger.warning("NewsData.io API key not configured")
            return []
        
        params = {
            "apikey": self.newsdata_api_key,
            "q": question,
            "language": "en",
            "size": 5,
            "category": "education"
        }
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.newsdata_url,
                        params=params,
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("results", [])
                            
                            snippets = []
                            for result in results:
                                title = result.get("title", "")
                                description = result.get("description", "")
                                if title and description:
                                    formatted_snippet = f"{title}: {description}"
                                    snippets.append(formatted_snippet)
                            
                            return snippets
                        else:
                            logger.warning(f"NewsData.io returned status {response.status}")
                            return []
        except Exception as e:
            logger.error(f"NewsData.io request failed: {e}")
            return []
    
    async def _calculate_relevance(self, query: str, snippet: str) -> float:
        """
        Calculate relevance score between query and snippet using keyword matching and semantic similarity.
        Returns float between 0.0 and 1.0.
        """
        try:
            # Convert to lowercase for comparison
            query_lower = query.lower()
            snippet_lower = snippet.lower()
            
            # Extract key terms from query
            query_terms = set(query_lower.split())
            snippet_terms = set(snippet_lower.split())
            
            # Remove common stop words
            stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'an', 'have', 'in', 'be', 'of', 'for', 'with', 'by', 'from', 'about', 'this', 'that', 'it', 'or', 'but', 'will', 'can', 'should', 'would', 'could', 'may', 'might'}
            query_terms = query_terms - stop_words
            snippet_terms = snippet_terms - stop_words
            
            if not query_terms:
                return 0.5  # Default score if no meaningful terms
            
            # Calculate keyword overlap
            common_terms = query_terms.intersection(snippet_terms)
            keyword_score = len(common_terms) / len(query_terms)
            
            # Boost for educational keywords
            edu_keywords = {'university', 'college', 'institute', 'engineering', 'admission', 'course', 'program', 'fee', 'fees', 'placement', 'campus', 'department', 'faculty', 'student', 'academic', 'degree', 'btech', 'mtech', 'phd', 'research'}
            edu_boost = 0.0
            for term in query_terms:
                if term in edu_keywords:
                    if term in snippet_terms:
                        edu_boost += 0.2
            
            # Calculate position boost (terms appearing early get higher score)
            position_boost = 0.0
            snippet_words = snippet_lower.split()[:50]  # Check first 50 words
            for i, word in enumerate(snippet_words):
                if word in query_terms:
                    position_boost += (50 - i) / 50 * 0.1
            
            # Combine scores
            final_score = min(1.0, keyword_score + edu_boost + position_boost)
            return final_score
            
        except Exception as e:
            logger.warning(f"Relevance calculation failed: {e}")
            return 0.5  # Default score on error

    async def _generate_search_query(self, query: str) -> str:
        """Generate optimized search query for educational content."""
        # Extract college name and create focused search
        college_keywords = ["university", "college", "institute", "school"]
        educational_terms = ["courses", "admission", "fees", "programs", "engineering", "btech", "mtech"]
        
        if any(keyword in query.lower() for keyword in college_keywords):
            # Add educational context to the search
            enhanced_query = f"{query} courses admission programs engineering"
            return enhanced_query
        
        return query
    
    async def _add_to_chromadb(self, content_parts: List[str], sources: List[str]):
        """Add scraped content to ChromaDB for future use."""
        try:
            documents = []
            for i, content in enumerate(content_parts):
                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={"source": sources[i] if i < len(sources) else "Unknown"}
                    )
                    documents.append(doc)
            
            if documents:
                self.db.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to add documents to ChromaDB: {e}")
    
    async def _gemini_fallback_knowledge(self, query: str) -> str:
        """Use Gemini Pro's internal knowledge as fallback."""
        try:
            fallback_prompt = f"""You are a knowledgeable assistant about Indian universities and colleges. 
            Provide comprehensive information about: {query}
            
            Include details about:
            - Basic information and location
            - Academic programs offered
            - Notable features or specializations
            - General admission information
            
            Format your response clearly with bullet points."""
            
            response = await self.gemini_fallback.ainvoke(fallback_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the response
            cleaned_content = content.replace("Response to:", "").strip()
            return cleaned_content if cleaned_content else ""
            
        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            return ""
