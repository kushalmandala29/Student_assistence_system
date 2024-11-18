import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use the latest embedder model
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# Create or load the vector database
db = Chroma(persist_directory="./db-hormozi", embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Use Llama 3.2 as the language model
local_llm = 'llama3.2'

llm = ChatOllama(model=local_llm, keep_alive="3h", max_tokens=512, temperature=0)

template = """<bos><start_of_turn>user\nYou are a Student Assistant System designed to help students with information regarding colleges, universities, ug and pg programmes with branch ,courses including each specializations, and their careers. \
Please provide a meaningful response based on the following context. Ensure your answer is in full sentences with correct spelling and punctuation. \
If the context does not contain the answer, respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Function to ask questions
def ask_question(question):
    print("Answer:\n\n", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")