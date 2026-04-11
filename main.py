import os
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def main():
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain AI Agents in 3 lines",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    print(chat_completion.choices[0].message.content)

asyncio.run(main())