import os
import asyncio
from dotenv import load_dotenv
from typing import Any, Dict

# Import ADK & Gemini libraries
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Load .env file
load_dotenv()


try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Gemini API Key not found in .env file.")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print(" Gemini API key setup complete.")
except Exception as e:
    print(f" Authentication Error: {e}")
    exit()

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Global configuration
APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Create or get session
    try:
        session = await session_service.create_session(
            app_name=runner_instance.app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception:
        session = await session_service.get_session(
            app_name=runner_instance.app_name, user_id=USER_ID, session_id=session_name
        )

    # Convert single query into list
    if user_queries and isinstance(user_queries, str):
        user_queries = [user_queries]

    if not user_queries:
        print("No queries provided.")
        return

    # Iterate through user queries
    for query in user_queries:
        print(f"\nUser > {query}")
        content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream the response asynchronously
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=content
        ):
            if event.content and event.content.parts:
                response = event.content.parts[0].text
                if response and response != "None":
                    print(f"{MODEL_NAME} > {response}")


async def in_memory_session_demo():
    print("\n🚀 Running In-Memory Session Demo...")

    root_agent = Agent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A simple text chatbot.",
    )

    global session_service
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    await run_session(
        runner,
        [
            "Hi, I am Sam! What is the capital of the United States?",
            "Hello! What is my name?",
        ],
        "stateful-agentic-session",
    )


async def persistent_session_demo():
    print("\n💾 Running Persistent Database Session Demo...")

    chatbot_agent = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A chatbot with persistent memory.",
    )

    db_url = "sqlite:///my_agent_data.db"
    global session_service
    session_service = DatabaseSessionService(db_url=db_url)
    runner = Runner(agent=chatbot_agent, app_name=APP_NAME, session_service=session_service)

    await run_session(
        runner,
        [
            "Hi, I am Sam! What is the capital of the United States?",
            "Hello! What is my name?",
        ],
        "test-db-session-01",
    )

    # Re-run to test persistence
    await run_session(
        runner,
        ["What is the capital of India?", "What is my name?"],
        "test-db-session-01",
    )



async def context_compaction_demo():
    print("\n🧠 Running Context Compaction Demo...")

    chatbot_agent = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A chatbot with context compaction.",
    )

    research_app = App(
        name="research_app_compacting",
        root_agent=chatbot_agent,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3, overlap_size=1
        ),
    )

    db_url = "sqlite:///my_agent_data.db"
    global session_service
    session_service = DatabaseSessionService(db_url=db_url)
    runner = Runner(app=research_app, session_service=session_service)


    await run_session(
        runner, "What is the latest news about AI in healthcare?", "compaction_demo"
    )
    await run_session(runner, "Are there any new developments in drug discovery?", "compaction_demo")
    await run_session(runner, "Tell me more about drug discovery.", "compaction_demo")
    await run_session(runner, "Who are the main companies in this field?", "compaction_demo")



def save_userinfo(tool_context: ToolContext, user_name: str, country: str) -> Dict[str, Any]:
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    return {"status": "success"}


def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    user_name = tool_context.state.get("user:name", "Unknown")
    country = tool_context.state.get("user:country", "Unknown")
    return {"user_name": user_name, "country": country}


async def session_state_demo():
    print("\n👤 Running Session State Demo...")

    root_agent = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A chatbot with session state management.",
        tools=[save_userinfo, retrieve_userinfo],
    )

    global session_service
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name=APP_NAME)

    await run_session(
        runner,
        [
            "Hi there, what is my name?",
            "My name is Sam. I'm from Poland.",
            "What is my name? Which country am I from?",
        ],
        "state-demo-session",
    )



if __name__ == "__main__":
    asyncio.run(in_memory_session_demo())
    asyncio.run(persistent_session_demo())
    asyncio.run(context_compaction_demo())
    asyncio.run(session_state_demo())
