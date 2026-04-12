import os
import asyncio
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

with open("prompt_library.json") as f:
    library = json.load(f)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

async def test(name, template, text):
    token_count = len(enc.encode(template["user"].format(text=text)))
    print(f"[{name}] Token count: {token_count}")
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(text=text)}
        ],
        model="llama-3.3-70b-versatile"
    )
    print(f"\n[{name}]")
    print(response.choices[0].message.content)
    print("-" * 40)

async def main():
    sample = "Python is a programming language used in AI, web development, and data science."
    for name, template in library.items():
        await test(name, template, sample)

asyncio.run(main())