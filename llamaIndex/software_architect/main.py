from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine




cleanArchitecureDocument = SimpleDirectoryReader("data/clean_architecture").load_data()
DDIADocument = SimpleDirectoryReader("data/DDIA").load_data()


# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3", request_timeout=30.0)



cadIndex = VectorStoreIndex.from_documents(
    cleanArchitecureDocument,
)

ddiaIndex = VectorStoreIndex.from_documents(
    DDIADocument,
)


cadQueryEngineTool = QueryEngineTool(
        query_engine=cadIndex.as_query_engine(),
        metadata=ToolMetadata(
            name="clean software architecture ",
            description="Provides information about software architecture fundamentals, what is good architectre and what is bad architecture",
        ),
    );


ddiaQueryEnginTool = QueryEngineTool(
        query_engine=ddiaIndex.as_query_engine(),
        metadata=ToolMetadata(
            name="designing high scalable robust softwares",
            description="describes details about building high performance softwares, details about respective technologies and their trade offs",
        ),
    );


query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[cadQueryEngineTool, ddiaQueryEnginTool]
)


while(query := input("Enter a query (q to quit): ")) != "q":
    response = query_engine.query(query)
    print(response)
    
