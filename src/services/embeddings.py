import base64
import logging
from urllib.parse import urlparse

import pypandoc
import regex
from typing import Any, Dict, Optional

from firecrawl import Firecrawl
from firecrawl.v2.utils.error_handler import BadRequestError
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.services.vector_store import get_vector_store
from src.shared.constants import (
    INVALID_UNICODE_CLEANUP_REGEX,
    VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD,
    VECTOR_EMBEDDINGS_QUERY_SYSTEM_PROMPT
)
from src.shared.enums import DocType, SourceType
from src.shared.schemas import DocumentData, QAPair

logger = logging.getLogger(__name__)


class InvalidURLError(ValueError):
    """Custom exception for invalid URLs provided for scraping."""
    pass


def _sanitize_for_doc_id(text: str) -> str:
    """Sanitizes a string to be used as a document ID."""
    return regex.sub(r'[^a-zA-Z0-9]+', '_', text.lower()).strip('_')


def store_data_from_qa_pair(qa_pair: QAPair, practice_id: str):
    """
    Stores a Q&A pair in Chroma.
    """
    vector_store = get_vector_store()

    doc_id = _sanitize_for_doc_id(qa_pair.question)

    try:
        logger.info(f"Checking for existing document with doc_id '{doc_id}' for practice_id: {practice_id}...")
        existing_docs = vector_store.get(
            where={
                "$and": [
                    {"practice_id": practice_id},
                    {"source_type": SourceType.QA_PAIR.value},
                    {"doc_id": doc_id}
                ]
            },
            include=[]
        )
        existing_ids = existing_docs.get("ids", [])

        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing documents for Q&A pair. Deleting them before adding new document...")
            vector_store.delete(ids=existing_ids)
            logger.info(f"Successfully deleted {len(existing_ids)} existing documents for Q&A pair.")
        else:
            logger.info(f"No existing document found for Q&A pair with doc_id '{doc_id}'.")
    except Exception as e:
        logger.error(f"Error while checking/deleting existing documents for Q&A pair: {e}", exc_info=True)
        raise

    content = f"Q: {qa_pair.question}\nA: {qa_pair.answer}"
    doc = Document(page_content=content)
    doc.metadata["doc_id"] = doc_id
    doc.metadata["practice_id"] = practice_id
    doc.metadata["source_type"] = SourceType.QA_PAIR.value

    try:
        logger.info(f"Adding new Q&A pair document to vector store with ID {doc_id}.")
        vector_store.add_documents(documents=[doc], ids=[doc_id])
        logger.info(
            f"Successfully added new Q&A pair from '{qa_pair.question}' to the collection."
        )
    except Exception as e:
        logger.error(f"Error adding Q&A pair document to vector store for practice_id {practice_id}: {e}", exc_info=True)
        raise


def store_data_from_document(document_data: DocumentData, practice_id: str):
    """
    Processes a document and stores its content in Chroma.
    """
    vector_store = get_vector_store()
    doc_id = _sanitize_for_doc_id(document_data.name)

    try:
        logger.info(f"Checking for existing document with doc_id '{doc_id}' for practice_id: {practice_id}...")
        existing_docs = vector_store.get(
            where={
                "$and": [
                    {"practice_id": practice_id},
                    {"source_type": SourceType.DOCUMENT.value},
                    {"doc_type": document_data.docType},
                    {"doc_id": doc_id}
                ]
            },
            include=[]
        )
        existing_ids = existing_docs.get("ids", [])

        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing documents for document. Deleting them before adding new document...")
            vector_store.delete(ids=existing_ids)
            logger.info(f"Successfully deleted {len(existing_ids)} existing documents for document.")
        else:
            logger.info(f"No existing document found for document with doc_id '{doc_id}'.")
    except Exception as e:
        logger.error(f"Error while checking/deleting existing documents for document: {e}", exc_info=True)
        raise

    try:
        decoded_data = base64.b64decode(document_data.data)
        if document_data.docType == DocType.DOCX:
            # Note: pypandoc requires pandoc to be installed on the system.
            content = pypandoc.convert_text(decoded_data, "markdown", format="docx")
        elif document_data.docType == DocType.TXT:
            content = decoded_data.decode('utf-8')
        else:
            logger.warning(f"Unsupported docType: {document_data.docType}. Skipping.")
            return
    except Exception as e:
        logger.error(f"Error processing document {document_data.name}: {e}", exc_info=True)
        raise

    cleaned_content = regex.sub(INVALID_UNICODE_CLEANUP_REGEX, '', content)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=128,
    )
    docs = text_splitter.create_documents([cleaned_content])
    logger.info(f"Split content from {document_data.name} into {len(docs)} documents.")

    ids = []
    for i, doc in enumerate(docs):
        chunk_id = f"{doc_id}_{i}"
        doc.metadata["doc_id"] = doc_id
        doc.metadata["practice_id"] = practice_id
        doc.metadata["source_type"] = SourceType.DOCUMENT.value
        ids.append(chunk_id)

    try:
        logger.info(f"Adding {len(docs)} new document chunks to vector store.")
        vector_store.add_documents(documents=docs, ids=ids)
        logger.info(
            f"Successfully added {len(docs)} new chunks from {document_data.name} to the collection."
        )
    except Exception as e:
        logger.error(f"Error adding document chunks to vector store for practice_id {practice_id}: {e}", exc_info=True)
        raise


def store_data_from_website(website: str, practice_id: str):
    """
    Scrapes a website and stores its content in Chroma.
    """
    if not settings.FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY not found in settings")

    vector_store = get_vector_store()
    firecrawl = Firecrawl(
        api_key=settings.FIRECRAWL_API_KEY,
    )

    parsed_url = urlparse(website)
    endpoint = parsed_url.netloc + parsed_url.path
    sanitized_url = _sanitize_for_doc_id(endpoint)

    try:
        logger.info(f"Checking for existing documents with URL prefix '{sanitized_url}' for practice_id: {practice_id}...")
        existing_docs = vector_store.get(
            where={
                "$and": [
                    {"practice_id": practice_id},
                    {"source_type": SourceType.WEB_PAGE.value},
                    {"source_url": website}
                ]
            },
            include=[]
        )
        existing_ids = existing_docs.get("ids", [])
        
        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing documents for URL {website}. Deleting them before adding new chunks...")
            vector_store.delete(ids=existing_ids)
            logger.info(f"Successfully deleted {len(existing_ids)} existing chunks for {website}.")
        else:
            logger.info(f"No existing documents found for URL {website}.")
    except Exception as e:
        logger.error(f"Error while checking/deleting existing documents for {website}: {e}", exc_info=True)
        raise

    logger.info(f"Scraping {website} for practice_id: {practice_id}...")
    try:
        scraped_website = firecrawl.scrape(
            url=website,
            formats=["markdown"],
            exclude_tags=
                ["script", "style", "img", "a", "source", "track", "embed", "base", "col", "area", "form", "input"],
        )
    except BadRequestError as e:
        logger.warning(f"Firecrawl failed to scrape URL {website} due to a bad request: {e}")
        raise InvalidURLError(f"The URL '{website}' is invalid or could not be scraped.") from e
    output_markdown = scraped_website.markdown
    if not output_markdown:
        logger.warning(f"No markdown content scraped from {website}. Skipping.")
        return
        
    cleaned_markdown = regex.sub(INVALID_UNICODE_CLEANUP_REGEX, '', output_markdown)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=128,
    )
    docs = text_splitter.create_documents([cleaned_markdown])
    logger.info(f"Split content from {website} into {len(docs)} documents.")

    ids = []
    for i, doc in enumerate(docs):
        doc_id = f"{sanitized_url}_{i}"
        doc.metadata["doc_id"] = doc_id
        doc.metadata["practice_id"] = practice_id
        doc.metadata["source_type"] = SourceType.WEB_PAGE.value
        doc.metadata["source_page_title"] = getattr(scraped_website.metadata, 'title', 'No Title')
        doc.metadata["source_url"] = website
        ids.append(doc_id)

    try:
        logger.info(f"Adding {len(docs)} new documents to vector store.")
        vector_store.add_documents(documents=docs, ids=ids)
        logger.info(
            f"Successfully added {len(docs)} new chunks from {website} to the collection."
        )
    except Exception as e:
        logger.error(f"Error adding documents to vector store for website {website} and practice_id {practice_id}: {e}", exc_info=True)
        raise


def delete_data_from_document(document_name: str, practice_id: str) -> int:
    """
    Deletes all chunks of a document from Chroma based on the document name and practice ID.
    Returns the number of documents deleted.
    """
    vector_store = get_vector_store()
    doc_id = _sanitize_for_doc_id(document_name)

    try:
        logger.info(f"Searching for document to delete with doc_id: {doc_id} and practice_id: {practice_id}...")
        existing_docs = vector_store.get(
            where={
                "$and": [
                    {"practice_id": practice_id},
                    {"source_type": SourceType.DOCUMENT.value},
                    {"doc_id": doc_id}
                ]
            },
            include=[]
        )
        existing_ids = existing_docs.get("ids", [])

        if existing_ids:
            logger.info(f"Found {len(existing_ids)} documents for document. Deleting them...")
            vector_store.delete(ids=existing_ids)
            logger.info(f"Successfully deleted {len(existing_ids)} chunks for document.")
            return len(existing_ids)
        else:
            logger.info(f"No existing documents found for document.")
            return 0
    except Exception as e:
        logger.error(f"Error while deleting document chunks: {e}", exc_info=True)
        raise


def delete_data_from_qa_pair(question: str, practice_id: str) -> int:
    """
    Deletes a Q&A pair from Chroma based on the question and practice ID.
    Returns the number of documents deleted.
    """
    vector_store = get_vector_store()
    doc_id = _sanitize_for_doc_id(question)

    try:
        logger.info(f"Searching for Q&A pair to delete with doc_id: {doc_id} and practice_id: {practice_id}...")
        existing_docs = vector_store.get(
            where={
                "$and": [
                    {"practice_id": practice_id},
                    {"source_type": SourceType.QA_PAIR.value},
                    {"doc_id": doc_id}
                ]
            },
            include=[]
        )
        existing_ids = existing_docs.get("ids", [])

        if existing_ids:
            logger.info(f"Found {len(existing_ids)} documents for Q&A pair. Deleting them...")
            vector_store.delete(ids=existing_ids)
            logger.info(f"Successfully deleted {len(existing_ids)} chunks for Q&A pair.")
            return len(existing_ids)
        else:
            logger.info(f"No existing documents found for Q&A pair.")
            return 0
    except Exception as e:
        logger.error(f"Error while deleting Q&A pair documents: {e}", exc_info=True)
        raise


def delete_data_from_website(website: str, practice_id: str) -> int:
    """
    Deletes all documents from Chroma that are associated with a specific website URL and practice ID.
    Returns the number of documents deleted.
    """
    vector_store = get_vector_store()

    try:
        logger.info(f"Searching for documents to delete for URL: {website} and practice_id: {practice_id}...")
        existing_docs = vector_store.get(
            where={
                "$and": [
                    {"practice_id": practice_id},
                    {"source_type": SourceType.WEB_PAGE.value},
                    {"source_url": website}
                ]
            },
            include=[]
        )
        existing_ids = existing_docs.get("ids", [])

        if existing_ids:
            logger.info(f"Found {len(existing_ids)} documents for URL {website}. Deleting them...")
            vector_store.delete(ids=existing_ids)
            logger.info(f"Successfully deleted {len(existing_ids)} chunks for {website}.")
            return len(existing_ids)
        else:
            logger.info(f"No existing documents found for URL {website}.")
            return 0
    except Exception as e:
        logger.error(f"Error while deleting documents for {website}: {e}", exc_info=True)
        raise


def retrieve_data(query: str, practice_id: str, filters: Optional[Dict[str, Any]] = None) -> tuple[str, bool]:
    """
    Retrieves data from the vector store based on a query and optional filters,
    and generates a response using an LLM.

    Args:
        query: The user's question.
        practice_id: The practice ID to filter the search results.
        filters: A dictionary of metadata to filter the search results.

    Returns:
        A tuple containing:
        - The content of the model's response (str).
        - A boolean indicating if relevant data was found (bool).
    """
    vector_store = get_vector_store()

    search_filters = filters.copy() if filters else {}
    search_filters["practice_id"] = practice_id

    results_with_scores = vector_store.similarity_search_with_score(
        query=query, k=3, filter=search_filters
    )

    if not results_with_scores:
        logger.warning(f"No results found for query: '{query}' with filters: {search_filters}")
        return "No relevant information was found to answer your question.", False

    filtered_results_with_scores = [
        (doc, score) for doc, score in results_with_scores if score < VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD
    ]

    if filtered_results_with_scores:
        source_types_present = {doc.metadata.get("source_type") for doc, _ in filtered_results_with_scores}

        priority_order = [SourceType.QA_PAIR.value, SourceType.WEB_PAGE.value, SourceType.DOCUMENT.value]

        for source_type in priority_order:
            if source_type in source_types_present:
                prioritized_results = [
                    (doc, score) for doc, score in filtered_results_with_scores
                    if doc.metadata.get("source_type") == source_type
                ]
                if prioritized_results:
                    logger.info(f"Found {len(prioritized_results)} results of priority type '{source_type}', prioritizing them.")
                    filtered_results_with_scores = prioritized_results
                    break

    logger.info(f"Found {len(filtered_results_with_scores)} results for query: '{query}'")
    for doc, score in filtered_results_with_scores:
        doc_id = doc.metadata.get('doc_id', 'N/A')
        logger.info(f"  - Document ID: {doc_id}, Score (distance): {score:.4f}")

    results = [
        doc for doc, _ in filtered_results_with_scores
    ]

    if not results:
        logger.warning(
            f"No results found within similarity threshold ({VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD}) for query: '{query}'"
        )
        return "No relevant information was found to answer your question.", False

    context = "\n---\n".join([doc.page_content for doc in results])
    prompt = ChatPromptTemplate.from_template(VECTOR_EMBEDDINGS_QUERY_SYSTEM_PROMPT)
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
    )

    chain = prompt | model

    response = chain.invoke({"context": context, "question": query})

    return response.content, True
