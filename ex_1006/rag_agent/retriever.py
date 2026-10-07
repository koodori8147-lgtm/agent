from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DB_PATH = str(Path(__file__).parent / "chroma_db")

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="report_pdf",
)
vectorstore.get()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="use this tool to search useful information from the stock report PDF document",
)