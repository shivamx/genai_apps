from getpass import getpass
import logging
import sys
from pprint import pprint

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser,SimpleNodeParser

#
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
#


documents = SimpleDirectoryReader('./Data/').load_data()
#print(len(documents))
#pprint(documents[5].metadata)

llm = Ollama(
    model="llama3",
    request_timeout=120.0,
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.0}
    )



##The choice between sending the entire Document object to the index 
## or converting the Document into Node objects before indexing depends on your specific use case and the structure of your data.

## Converting the Document into Node objects before indexing: This approach is practical when your documents are long, and you want to break them down into smaller chunks (or nodes) before indexing. 
##This can be beneficial when you want to retrieve specific parts of a document rather than the entire document.


##Surrounding Window context: how to build in llamaIndex:
## The SentenceWindowNodeParser class is designed to parse documents into nodes (sentences) and capture a window of surrounding sentences for each node.
##This can be useful for context-aware text processing, where understanding the surrounding context of a sentence can provide valuable insights.


sentence_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text")

base_node_parser = SimpleNodeParser()


nodes = sentence_node_parser.get_nodes_from_documents(documents)
base_nodes = base_node_parser.get_nodes_from_documents(documents)


#print(f"SENTENCE NODES :\n {nodes[10]}")
#print(f"BASE NODES :\n {base_nodes[10]}")
#print(dict(nodes[10]))


## An IndexNode is a node object used in LlamaIndex.
## It represents chunks of the original documents that are stored in an Index. 
## The Index is a data structure that allows for quick retrieval of relevant context for a user query, which is fundamental for retrieval-augmented generation (RAG) use cases.
## At its core, the IndexNode inherits properties from a TextNode, meaning it primarily represents textual content.
##However, the distinguishing feature of an IndexNode is its index_id attribute. This index_id acts as a unique identifier or reference to another object, 



## The ServiceContext is a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex pipeline/application.

## A VectorStoreIndex in LlamaIndex is a type of index that uses vector representations of text for efficient retrieval of relevant context.
## It is built on top of a VectorStore, which is a data structure that stores vectors and allows for quick nearest neighbor search.
##The VectorStoreIndex takes in IndexNode objects, which represent chunks of the original documents.
