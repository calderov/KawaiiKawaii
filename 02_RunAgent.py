import os

from dotenv import load_dotenv
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings as LlamaGlobalSettings

from EasySpeech import easySpeech

def InitializeAgent():
    # Load API keys
    load_dotenv()

    # Set the default model to use for embeddings
    LlamaGlobalSettings.embed_model = OpenAIEmbedding()

    # Create an llm object to use for the QueryEngine and the ReActAgent
    llm = OpenAI(
        model="gpt-3.5-turbo",
        context_window=10000,
        is_function_calling_model=True,
        is_chat_model=True,
    )

    # Load data from vector database
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    storage_index = load_index_from_storage(storage_context)
    storage_query_engine = storage_index.as_query_engine(similarity_top_k=3, llm=llm)

    # Create agent to respond queries based on data in vector database
    query_engine_tools = [
        QueryEngineTool(
            query_engine=storage_query_engine,
            metadata=ToolMetadata(
                name="octoai_company_data",
                description="OctoAI company profile (scope, services, history, role descriptions, etc)",
            ),
        ),
    ]

    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        max_turns=10, # max iterations?
    )

    return agent

def PrintAndSay(text, subject=""):
    print(subject, ">>>", text)
    easySpeech.TextToSpeechNatural(text)

if __name__ == "__main__":
    agent = InitializeAgent()

    # 0. KawaiiKawaii introduces itself   
    agent_response = agent.chat(
        """
        You are KawaiiKawaii, a cute and helpful assistant, your user Sofia Bennet is your user, please introduce yourself.
        """
    )
    input("KawaiiKawaii is ready...")
    PrintAndSay(str(agent_response), "KawaiiKawaii")

    # 2. Human responds to the introduction
    # TODO: Make this happen through speech recognition (whisper, speech_recognition, etc)
    human_response = "Hey, I'm planning a trip to Barcelona, can you suggest places and activities that I may like"
    input(">>")

    # 3. KawaiiKawaii follows up on human response and ask another question
    agent_response = agent.chat(
       f"""
        You just introduced yourself, Sofia Bennet your user responded the following:
        {human_response}
        ----
        Please, respond to Sofia based on her input and on the information you have about her preferences, age, previous travel destinations, etc.
        """
    )
    PrintAndSay(str(agent_response), "KawaiiKawaii")

    # 4. Human responds to the suggestions
    # TODO: Make this happen through speech recognition (whisper, speech_recognition, etc)
    human_response = "Thanks, I'm visiting Milan afterwards, I should be there by November 15th. Please remind me my activities in Milan and recomend accessories of brands I like"
    input(">>")

    # 5. KawaiiKawaii follows up on human response and ask another question
    agent_response = agent.chat(
       f"""
        Sofia Bennet responded the following:
        {human_response}
        ----
        Please, respond to Sofia based on her input and on the information you have about her preferences, age, previous travel destinations. Also remind her that she has an alarm set in 5 minutes to subscribe to Phoenix.
        """
    )
    PrintAndSay(str(agent_response), "KawaiiKawaii")