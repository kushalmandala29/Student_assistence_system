"""
Student Assistant RAG System - Main Application
Provides comprehensive college and university information using RAG pipeline.
"""

import streamlit as st
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag_pipeline.retriever import DocumentRetriever
from rag_pipeline.generator import ResponseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StudentAssistantApp:
    """Main application class for Student Assistant RAG system."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        try:
            self.retriever = DocumentRetriever()
            self.generator = ResponseGenerator()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            st.error("Failed to initialize the system. Please check your API keys and try again.")
            st.stop()
    
    async def process_query(self, query: str) -> str:
        """Process student query through the RAG pipeline."""
        try:
            # Step 1: Retrieve context
            with st.spinner("🔍 Searching for relevant information..."):
                context, sources = await self.retriever.retrieve_context(query)
            
            # Step 2: Generate response
            with st.spinner("🤖 Generating comprehensive answer..."):
                response_data = await self.generator.generate_response(
                    query=query,
                    context=context,
                    sources=sources
                )
            
            # Step 3: Format final response
            formatted_response = self.generator.format_final_response(response_data)
            
            # Log successful processing
            logger.info(f"Successfully processed query: {query[:50]}...")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again or check the official university website for information."
    
    def setup_streamlit_ui(self):
        """Setup Streamlit user interface."""
        # Page configuration
        st.set_page_config(
            page_title="Student Assistant System",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better appearance
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-box {
            background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #b3d9ff;
            margin: 0.5rem 0;
            color: #1f77b4;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("# 🎓 Student Assistant")
            st.markdown("**Your AI-powered guide to Indian colleges and universities**")
            
            st.markdown("### 🚀 Features")
            st.markdown("""
            <div class="feature-box">
            • Comprehensive college information<br>
            • Real-time web search<br>
            • Admission guidance<br>
            • Course details & career paths<br>
            • Smart relevance ranking
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📚 Ask about:")
            st.markdown("""
            - College details and rankings
            - Admission procedures
            - Course offerings and specializations
            - Fee structures and scholarships
            - Campus facilities and placements
            - Eligibility criteria
            """)
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        # Main content area
        st.markdown('<div class="main-header">🎓 Student Assistant System</div>', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "Hello! I'm your Student Assistant. I can help you find information about colleges, universities, courses, admissions, and more. What would you like to know?"
                }
            ]
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me anything about colleges, courses, or admissions...")
        
        # Process user input
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                # Process query asynchronously
                response = asyncio.run(self.process_query(user_input))
                st.markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    def run(self):
        """Run the Streamlit application."""
        try:
            self.setup_streamlit_ui()
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("An error occurred while running the application.")

def main():
    """Main function to run the Student Assistant System."""
    try:
        # Check for required environment variables
        required_env_vars = ["GOOGLE_GEMINI_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            st.info("Please set up your .env file with the required API keys.")
            st.stop()
        
        # Initialize and run the application
        app = StudentAssistantApp()
        app.run()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        st.error("Failed to start the application. Please check the logs and try again.")

if __name__ == "__main__":
    main()