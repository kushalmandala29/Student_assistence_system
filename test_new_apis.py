#!/usr/bin/env python3
"""
Test script for the new API integrations in Student Assistant RAG system.
Tests WebSearchAPI.ai and NewsData.io integrations.
"""

import asyncio
import os
from dotenv import load_dotenv
from rag_pipeline.retriever import DocumentRetriever

load_dotenv()

async def test_retrieval_pipeline():
    """Test the complete retrieval pipeline with fallback chain."""
    print("🧪 Testing Student Assistant RAG System with New APIs")
    print("=" * 60)
    
    # Initialize retriever
    retriever = DocumentRetriever()
    
    # Test queries
    test_queries = [
        "What are the courses offered at IIT Delhi?",
        "VIT University admission process",
        "Latest education news about engineering admissions"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 40)
        
        try:
            context, sources = await retriever.retrieve_context(query)
            
            print(f"📊 Context Length: {len(context)} characters")
            print(f"📚 Sources Found: {len(sources)}")
            
            if sources:
                print("🔍 Source Types:")
                for source in sources:
                    if "ChromaDB" in source:
                        print("  ✓ ChromaDB (Internal Knowledge)")
                    elif "WebSearchAPI" in source:
                        print("  ✓ WebSearchAPI.ai (Web Search)")
                    elif "NewsData.io" in source:
                        print("  ✓ NewsData.io (News Search)")
                    elif "Gemini Pro" in source:
                        print("  ✓ Gemini Pro (AI Fallback)")
            
            if context:
                preview = context[:200] + "..." if len(context) > 200 else context
                print(f"📄 Context Preview: {preview}")
            else:
                print("⚠️  No context retrieved")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 API Configuration Status:")
    print(f"  Gemini API: {'✓ Configured' if os.getenv('GOOGLE_GEMINI_API_KEY') else '❌ Missing'}")
    print(f"  WebSearchAPI: {'✓ Configured' if os.getenv('WEBSEARCH_API_KEY') else '❌ Missing'}")
    print(f"  NewsData.io: {'✓ Configured' if os.getenv('NEWSDATA_API_KEY') else '❌ Missing'}")
    
    print("\n💡 Note: To test WebSearchAPI and NewsData.io, add your API keys to .env file:")
    print("   WEBSEARCH_API_KEY=your_actual_key")
    print("   NEWSDATA_API_KEY=your_actual_key")

if __name__ == "__main__":
    asyncio.run(test_retrieval_pipeline())
