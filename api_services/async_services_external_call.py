from Anthropic import AsyncAnthropic
import asyncio

anthropic = AsyncAnthropic(
    api_key="[ENCRYPTION_KEY]", 
)

async def async_chat_completions():
    message = await anthropic.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        system="You are a helpful assistant.",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    )
    print(message.content)


asyncio.run(async_chat_completions())