import os
import asyncio
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def main():
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "List 3 Python tips as a JSON array. Format: [{\"tip\": \"...\", \"difficulty\": \"easy/medium/hard\"}]. Reply with JSON only, no extra text."
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    raw = response.choices[0].message.content.strip().strip("```").strip("json").strip()
    data = json.loads(raw)

    for item in data:
        print(f"Tip: {item['tip']}")
        print(f"Difficulty: {item['difficulty']}")
        print("---")

asyncio.run(main())