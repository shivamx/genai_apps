from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding


documents = SimpleDirectoryReader("data").load_data()

# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.llm = Ollama(model="llama3", request_timeout=30.0)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()


while(query := input("Enter a query (q to quit): ")) != "q":
    response = query_engine.query(query)
    print(response)