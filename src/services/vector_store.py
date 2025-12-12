from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import settings

_vector_store = None


def get_vector_store():
    """
    Returns a singleton instance of the Chroma vector store.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in settings")

    chroma_cloud_api_key = settings.CHROMA_CLOUD_API_KEY
    chroma_cloud_tenant = settings.CHROMA_CLOUD_TENANT
    chroma_cloud_database = settings.CHROMA_CLOUD_DATABASE
    chroma_cloud_collection = settings.CHROMA_CLOUD_COLLECTION

    if not all([chroma_cloud_api_key, chroma_cloud_tenant, chroma_cloud_database]):
        raise ValueError("One or more Chroma Cloud environment variables are not set in settings.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    _vector_store = Chroma(
        collection_name=chroma_cloud_collection,
        embedding_function=embeddings,
        chroma_cloud_api_key=chroma_cloud_api_key,
        tenant=chroma_cloud_tenant,
        database=chroma_cloud_database,
    )
    return _vector_store
