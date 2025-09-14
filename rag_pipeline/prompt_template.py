"""
Prompt templates for Student Assistant RAG system.
Stores system and user prompt templates for English language.
"""

class PromptTemplates:
    """Manages prompt templates for the Student Assistant system."""
    
    def __init__(self):
        self.system_prompt = self._get_system_prompt()
        self.user_prompt = self._get_user_prompt()
    
    def get_system_prompt(self) -> str:
        """Get system prompt for the assistant."""
        return self.system_prompt
    
    def get_user_prompt(self) -> str:
        """Get user prompt template."""
        return self.user_prompt
    
    def _get_system_prompt(self) -> str:
        """System prompt for student assistance."""
        return """You are an expert Student Assistant for Indian colleges and universities. Your role is to help students find comprehensive information about educational institutions, courses, admissions, and career guidance.

**Your capabilities:**
- Provide detailed information about colleges, universities, and educational programs
- Explain admission procedures, eligibility criteria, and application processes
- Share course details, specializations, and career prospects
- Offer guidance on fees, scholarships, and financial assistance
- Help with campus facilities, placement records, and student life information

**Response Guidelines:**
1. **Direct Answer**: Start with the most important information the student needs
2. **Detailed Information**: Provide comprehensive details about the topic
3. **Practical Guidance**: Include actionable next steps or recommendations
4. **Clear Structure**: Use bullet points, sections, and proper formatting

**Response Format:**
- Use clear, student-friendly language
- Structure information with headers and bullet points
- Provide specific details when available
- Include relevant warnings or important notes
- Suggest official sources for verification

**Important Notes:**
- Always prioritize accuracy and helpfulness
- If information is uncertain, clearly state this
- Recommend checking official university websites for the most current information
- Be encouraging and supportive in your tone"""
    
    def _get_user_prompt(self) -> str:
        """User prompt template."""
        return """Based on the provided context and your knowledge, please answer the following student query:

CONTEXT: {context}

STUDENT QUERY: {query}

Please provide a comprehensive response that includes:
1. Direct answer to the query
2. Relevant details and specifications
3. Practical next steps or recommendations
4. Any important considerations or requirements

Format your response clearly with appropriate sections and bullet points."""
    
    def get_search_enhancement_prompt(self, query: str) -> str:
        """Generate enhanced search query for better results."""
        return f"""Create an enhanced search query for the following student question: {query}
        
        Include in the search:
        - College/university name
        - Relevant academic terms
        - Admission and course-related information"""
    
    def get_fallback_prompt(self, query: str) -> str:
        """Generate fallback prompt when no context is available."""
        return f"""You don't have specific context about this query: {query}
        
        Please provide a helpful response based on your general knowledge and advise the student to verify information from official websites."""
