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
parser = StrOutputParser()

# Step 1: Generate outline
outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You write blog post outlines. Reply with exactly 3 section titles, numbered."),
    ("user", "Topic: {topic}")
])

# Step 2: Write intro FROM that outline
intro_prompt = ChatPromptTemplate.from_messages([
    ("system", "You write engaging blog introductions in 2 sentences."),
    ("user", "Write an intro for a blog post with this outline:\n{outline}")
])

outline_chain = outline_prompt | llm | parser
intro_chain = intro_prompt | llm | parser

# Step 1 output feeds Step 2
full_chain = outline_chain | (lambda outline: intro_chain.invoke({"outline": outline}))

result = full_chain.invoke({"topic": "Why AI Agents will replace traditional software"})
print(result)