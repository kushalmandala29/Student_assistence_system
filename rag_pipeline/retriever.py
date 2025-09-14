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
        Main retrieval method with fallback chain:
        ChromaDB → WebSearchAPI → Search1API → NewsData.io → Gemini Pro
        Returns (context, sources) tuple.
        """
        sources = []
        context_parts = []
        
        # Step 1: Try ChromaDB
        try:
            chroma_context, chroma_sources = await self._search_chromadb(query, threshold)
            if chroma_context:
                context_parts.append(chroma_context)
                sources.extend(chroma_sources)
                logger.info("Successfully retrieved context from ChromaDB")
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
        
        # Step 2: Try WebSearchAPI if ChromaDB insufficient
        if len(context_parts) == 0 or len(" ".join(context_parts)) < 200:
            try:
                websearch_snippets = await self.query_websearchapi(query)
                if websearch_snippets:
                    context_parts.extend(websearch_snippets)
                    sources.extend([f"WebSearchAPI - Result {i+1}" for i in range(len(websearch_snippets))])
                    logger.info("Successfully retrieved context from WebSearchAPI")
            except Exception as e:
                logger.warning(f"WebSearchAPI search failed: {e}")
        
        # Step 3: Try Search1API if still insufficient
        if len(context_parts) == 0 or len(" ".join(context_parts)) < 150:
            try:
                search1_snippets = await self.query_search1api(query)
                if search1_snippets:
                    context_parts.extend(search1_snippets)
                    sources.extend([f"Search1API - Result {i+1}" for i in range(len(search1_snippets))])
                    logger.info("Successfully retrieved context from Search1API")
            except Exception as e:
                logger.warning(f"Search1API search failed: {e}")
        
        # Step 4: Try NewsData.io if still insufficient
        if len(context_parts) == 0 or len(" ".join(context_parts)) < 100:
            try:
                news_snippets = await self.query_newsdata(query)
                if news_snippets:
                    context_parts.extend(news_snippets)
                    sources.extend([f"NewsData.io - Article {i+1}" for i in range(len(news_snippets))])
                    logger.info("Successfully retrieved context from NewsData.io")
            except Exception as e:
                logger.warning(f"NewsData.io search failed: {e}")
        
        # Step 5: Gemini Pro fallback if all fail
        if len(context_parts) == 0:
            try:
                gemini_context = await self._gemini_fallback_knowledge(query)
                if gemini_context:
                    context_parts.append(gemini_context)
                    sources.append("Gemini Pro")
                    logger.info("Using Gemini Pro fallback knowledge")
            except Exception as e:
                logger.error(f"Gemini Pro fallback failed: {e}")
        
        final_context = "\n\n".join(context_parts) if context_parts else ""
        return final_context, sources
    
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
