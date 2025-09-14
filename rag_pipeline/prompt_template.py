"""
Prompt templates for Student Assistant RAG system.
Stores system and user prompt templates with multilingual support.
"""

class PromptTemplates:
    """Manages prompt templates for different languages and use cases."""
    
    def __init__(self):
        self.system_prompts = {
            'en': self._get_english_system_prompt(),
            'hi': self._get_hindi_system_prompt()
        }
        
        self.user_prompts = {
            'en': self._get_english_user_prompt(),
            'hi': self._get_hindi_user_prompt()
        }
    
    def get_system_prompt(self, language: str = 'en') -> str:
        """Get system prompt for specified language."""
        return self.system_prompts.get(language, self.system_prompts['en'])
    
    def get_user_prompt(self, language: str = 'en') -> str:
        """Get user prompt template for specified language."""
        return self.user_prompts.get(language, self.user_prompts['en'])
    
    def _get_english_system_prompt(self) -> str:
        """English system prompt for student assistance."""
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
    
    def _get_hindi_system_prompt(self) -> str:
        """Hindi system prompt for student assistance."""
        return """आप भारतीय कॉलेजों और विश्वविद्यालयों के लिए एक विशेषज्ञ छात्र सहायक हैं। आपकी भूमिका छात्रों को शैक्षणिक संस्थानों, कोर्सेज, प्रवेश, और करियर मार्गदर्शन के बारे में व्यापक जानकारी प्रदान करना है।

**आपकी क्षमताएं:**
- कॉलेजों, विश्वविद्यालयों और शैक्षणिक कार्यक्रमों के बारे में विस्तृत जानकारी प्रदान करना
- प्रवेश प्रक्रियाओं, पात्रता मानदंडों और आवेदन प्रक्रियाओं की व्याख्या करना
- कोर्स विवरण, विशेषज्ञताओं और करियर संभावनाओं को साझा करना
- फीस, छात्रवृत्ति और वित्तीय सहायता पर मार्गदर्शन प्रदान करना
- कैंपस सुविधाओं, प्लेसमेंट रिकॉर्ड और छात्र जीवन की जानकारी में सहायता करना

**उत्तर दिशानिर्देश:**
1. **सीधा उत्तर**: छात्र को सबसे महत्वपूर्ण जानकारी के साथ शुरुआत करें
2. **विस्तृत जानकारी**: विषय के बारे में व्यापक विवरण प्रदान करें
3. **व्यावहारिक मार्गदर्शन**: कार्यात्मक अगले कदम या सिफारिशें शामिल करें
4. **स्पष्ट संरचना**: बुलेट पॉइंट्स, सेक्शन और उचित फॉर्मेटिंग का उपयोग करें

**महत्वपूर्ण नोट्स:**
- हमेशा सटीकता और सहायकता को प्राथमिकता दें
- यदि जानकारी अनिश्चित है, तो इसे स्पष्ट रूप से बताएं
- नवीनतम जानकारी के लिए आधिकारिक विश्वविद्यालय वेबसाइटों की जांच करने की सिफारिश करें
- अपने स्वर में उत्साहजनक और सहायक बनें"""
    
    def _get_english_user_prompt(self) -> str:
        """English user prompt template."""
        return """Based on the provided context and your knowledge, please answer the following student query:

CONTEXT: {context}

STUDENT QUERY: {query}

Please provide a comprehensive response that includes:
1. Direct answer to the query
2. Relevant details and specifications
3. Practical next steps or recommendations
4. Any important considerations or requirements

Format your response clearly with appropriate sections and bullet points."""
    
    def _get_hindi_user_prompt(self) -> str:
        """Hindi user prompt template."""
        return """प्रदान किए गए संदर्भ और आपके ज्ञान के आधार पर, कृपया निम्नलिखित छात्र प्रश्न का उत्तर दें:

संदर्भ: {context}

छात्र प्रश्न: {query}

कृपया एक व्यापक उत्तर प्रदान करें जिसमें शामिल हो:
1. प्रश्न का सीधा उत्तर
2. प्रासंगिक विवरण और विशिष्टताएं
3. व्यावहारिक अगले कदम या सिफारिशें
4. कोई महत्वपूर्ण विचारणीय बातें या आवश्यकताएं

अपने उत्तर को उपयुक्त सेक्शन और बुलेट पॉइंट्स के साथ स्पष्ट रूप से प्रारूपित करें।"""
    
    def get_search_enhancement_prompt(self, query: str, language: str = 'en') -> str:
        """Generate enhanced search query for better results."""
        if language == 'hi':
            return f"""निम्नलिखित छात्र प्रश्न के लिए एक बेहतर खोज क्वेरी बनाएं: {query}
            
            खोज में शामिल करें:
            - कॉलेज/विश्वविद्यालय का नाम
            - प्रासंगिक शैक्षणिक शब्द
            - प्रवेश और कोर्स संबंधी जानकारी"""
        else:
            return f"""Create an enhanced search query for the following student question: {query}
            
            Include in the search:
            - College/university name
            - Relevant academic terms
            - Admission and course-related information"""
    
    def get_fallback_prompt(self, query: str, language: str = 'en') -> str:
        """Generate fallback prompt when no context is available."""
        if language == 'hi':
            return f"""आपके पास इस प्रश्न के बारे में कोई विशिष्ट संदर्भ नहीं है: {query}
            
            कृपया अपने सामान्य ज्ञान के आधार पर एक सहायक उत्तर प्रदान करें और छात्र को आधिकारिक वेबसाइटों से जानकारी की पुष्टि करने की सलाह दें।"""
        else:
            return f"""You don't have specific context about this query: {query}
            
            Please provide a helpful response based on your general knowledge and advise the student to verify information from official websites."""
