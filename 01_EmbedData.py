import os
import shutil

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import Settings as LlamaGlobalSettings

def EmbedDataInVectorDatabase():
    # Load API keys
    load_dotenv()

    # Set the default model to use for embeddings
    LlamaGlobalSettings.embed_model = OpenAIEmbedding()

    # Remove vector database if already existing (this will force a fresh database)
    if os.path.exists("./storage"):
        shutil.rmtree("./storage")

    # Load documents
    documents = SimpleDirectoryReader(
        input_files=[
            "./data/user_profile.md"
        ]
    ).load_data()

    # Build index
    storage_index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Persist vector database
    storage_index.storage_context.persist(persist_dir="./storage")

if __name__ == "__main__":
    EmbedDataInVectorDatabase()