"""
Generator module for Student Assistant RAG system.
Calls Gemini model with retrieved context and formats responses.
"""

import os
import logging
from typing import Dict, List, Optional
from google_gemini import ChatGoogleGemini
from .prompt_template import PromptTemplates
import re

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates responses using Gemini model with retrieved context."""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        self.llm = ChatGoogleGemini(
            api_key=self.gemini_api_key,
            model="gemini-1.5-flash"
        )
        self.prompt_templates = PromptTemplates()
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text. Returns 'en' for English or detected language code."""
        # Simple language detection - can be enhanced with proper language detection library
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'about', 'university', 'college']
        hindi_indicators = ['के', 'में', 'है', 'का', 'की', 'से', 'यूनिवर्सिटी', 'कॉलेज', 'विश्वविद्यालय']
        
        text_lower = text.lower()
        
        english_count = sum(1 for word in english_indicators if word in text_lower)
        hindi_count = sum(1 for word in hindi_indicators if word in text_lower)
        
        if hindi_count > english_count:
            return 'hi'
        return 'en'
    
    async def generate_response(
        self, 
        query: str, 
        context: str, 
        sources: List[str],
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate formatted response with context and sources.
        
        Returns:
            Dict with 'answer', 'sources', and 'language' keys
        """
        try:
            # Auto-detect language if not provided
            if language is None:
                language = self.detect_language(query)
            
            # Get appropriate prompt template
            system_prompt = self.prompt_templates.get_system_prompt(language)
            
            # Format the prompt with context
            if context.strip():
                formatted_prompt = f"""{system_prompt}

CONTEXT: {context}

STUDENT QUERY: {query}

Please provide a comprehensive answer in {self._get_language_name(language)}."""
            else:
                formatted_prompt = f"""{system_prompt}

STUDENT QUERY: {query}

No specific context was found. Please provide a helpful general response about this topic in {self._get_language_name(language)}, and suggest checking official university websites for detailed information."""
            
            # Generate response
            response = await self.llm.ainvoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and format the answer
            formatted_answer = self._format_answer(answer, sources)
            
            return {
                'answer': formatted_answer,
                'sources': self._format_sources(sources),
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'answer': self._get_fallback_response(language),
                'sources': "Error occurred during processing",
                'language': language or 'en'
            }
    
    def _format_answer(self, answer: str, sources: List[str]) -> str:
        """Clean and format the generated answer."""
        # Remove unwanted prefixes
        unwanted_prefixes = [
            "Response to:",
            "Based on the provided context",
            "Based on the information provided",
            "According to the context",
            "I found information about"
        ]
        
        cleaned_answer = answer
        for prefix in unwanted_prefixes:
            if cleaned_answer.startswith(prefix):
                cleaned_answer = cleaned_answer[len(prefix):].strip()
        
        # Remove special tokens
        cleaned_answer = re.sub(r'<[^>]+>', '', cleaned_answer)
        cleaned_answer = cleaned_answer.replace("ANSWER:", "").strip()
        
        # Ensure proper formatting
        if not cleaned_answer:
            return "I couldn't find complete details. Please check the official university website."
        
        return cleaned_answer
    
    def _format_sources(self, sources: List[str]) -> str:
        """Format sources information."""
        if not sources:
            return "No sources available"
        
        # Categorize sources
        chromadb_sources = [s for s in sources if "ChromaDB" in s]
        websearch_sources = [s for s in sources if "WebSearchAPI" in s]
        duckduckgo_sources = [s for s in sources if "DuckDuckGo" in s]
        newsdata_sources = [s for s in sources if "NewsData.io" in s]
        gemini_sources = [s for s in sources if "Gemini Pro" in s]
        
        source_info = []
        
        if chromadb_sources:
            source_info.append("• Internal Knowledge Base")
        
        if websearch_sources:
            source_info.append("• Web Search Results (WebSearchAPI)")
            # Add specific URLs if available
            urls = [s.split(" - ")[-1] for s in websearch_sources if " - " in s]
            for url in urls[:2]:  # Limit to 2 URLs for readability
                if url.startswith("http"):
                    source_info.append(f"  - {url}")
        
        if duckduckgo_sources:
            source_info.append("• Search Results (DuckDuckGo)")
        
        if newsdata_sources:
            source_info.append("• News & Education Updates (NewsData.io)")
            # Add specific sources if available
            news_sources = [s.split(" - ")[-1] for s in newsdata_sources if " - " in s]
            for news_source in news_sources[:2]:  # Limit to 2 sources for readability
                if news_source and not news_source.startswith("http"):
                    source_info.append(f"  - {news_source}")
        
        if gemini_sources:
            source_info.append("• AI Knowledge Base (Gemini Pro)")
        
        return "\n".join(source_info) if source_info else "Multiple sources"
    
    def _get_language_name(self, language_code: str) -> str:
        """Convert language code to readable name."""
        language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'bn': 'Bengali'
        }
        return language_names.get(language_code, 'English')
    
    def _get_fallback_response(self, language: str) -> str:
        """Get fallback response when generation fails."""
        fallback_responses = {
            'en': "I couldn't find complete details about your query. Please check the official university website or contact their admissions office for accurate information.",
            'hi': "मुझे आपके प्रश्न के बारे में पूरी जानकारी नहीं मिल सकी। कृपया आधिकारिक विश्वविद्यालय वेबसाइट देखें या सटीक जानकारी के लिए उनके प्रवेश कार्यालय से संपर्क करें।"
        }
        return fallback_responses.get(language, fallback_responses['en'])
    
    def format_final_response(self, response_data: Dict[str, str]) -> str:
        """Format the final response for display."""
        answer = response_data.get('answer', '')
        sources = response_data.get('sources', '')
        
        if not answer:
            return "I couldn't find complete details. Please check the official university website."
        
        # Format with sections
        formatted_response = answer
        
        if sources and sources != "No sources available":
            formatted_response += f"\n\n**Sources:**\n{sources}"
        
        return formatted_response
