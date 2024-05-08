from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


load_dotenv()

chat = ChatOpenAI();

embeddings = OpenAIEmbeddings();
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)


retriever = db.as_retriever();

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever= retriever,
    chain_type="stuff"
)

result = chain.run("What is some fact about english language?");

print(result);

