import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class ConceptBreakdown(BaseModel):
    concept: str = Field(description="The concept name")
    simple_definition: str = Field(description="One sentence definition for a beginner")
    use_case: str = Field(description="One real-world use case")
    difficulty: str = Field(description="easy, medium, or hard")

parser = JsonOutputParser(pydantic_object=ConceptBreakdown)

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator. {format_instructions}"),
    ("user", "Break down this AI concept: {topic}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

for concept in ["RAG", "Vector Database", "AI Agent"]:
    result = chain.invoke({"topic": concept})
    print(f"\n📘 {result['concept']}")
    print(f"   Definition : {result['simple_definition']}")
    print(f"   Use Case   : {result['use_case']}")
    print(f"   Difficulty : {result['difficulty']}")
    print("-" * 50)