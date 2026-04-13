import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical explainer. Reply in 3 bullet points max."),
    ("user", "Explain this concept: {topic}")
])

parser = StrOutputParser()

# This pipe syntax IS LangChain
chain = prompt | llm | parser

result = chain.invoke({"topic": "LangChain"})
print(result)