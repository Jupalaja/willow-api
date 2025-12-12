import logging
from fastapi import APIRouter, HTTPException, status

from src.services.embeddings import (
    delete_data_from_document,
    delete_data_from_qa_pair,
    delete_data_from_website,
    store_data_from_document,
    store_data_from_qa_pair,
    store_data_from_website,
    InvalidURLError,
)
from src.shared.enums import SourceType
from src.shared.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    DeleteEmbeddingsRequest,
    DeleteEmbeddingsResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/embeddings", response_model=CreateEmbeddingsResponse)
async def create_embeddings(
    request: CreateEmbeddingsRequest,
):
    logger.info(f"Received create embeddings request: {request.model_dump_json(indent=2)}")

    if request.sourceType == SourceType.WEB_PAGE:
        if not request.sourceData.webPageURL:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="webPageURL is required for WEB_PAGE source type")
        try:
            store_data_from_website(request.sourceData.webPageURL, request.practiceId)
            return CreateEmbeddingsResponse(status="success", message="Embeddings created successfully from web page.")
        except InvalidURLError as e:
            logger.error(f"Failed to create embeddings from invalid web page {request.sourceData.webPageURL} for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The provided URL is invalid. Please check the URL and try again.")
        except Exception as e:
            logger.error(f"Failed to create embeddings from web page {request.sourceData.webPageURL} for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating embeddings from the web page.")
    elif request.sourceType == SourceType.QA_PAIR:
        if not request.sourceData.qa_pair:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="qa_pair is required for QA_PAIR source type")
        try:
            store_data_from_qa_pair(request.sourceData.qa_pair, request.practiceId)
            return CreateEmbeddingsResponse(status="success", message="Embeddings created successfully from Q&A pair.")
        except Exception as e:
            logger.error(f"Failed to create embeddings from Q&A pair for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating embeddings from the Q&A pair.")
    elif request.sourceType == SourceType.DOCUMENT:
        if not request.sourceData.document or not request.sourceData.document.data or not request.sourceData.document.docType or not request.sourceData.document.name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document with data, docType and name is required for DOCUMENT source type")
        try:
            store_data_from_document(request.sourceData.document, request.practiceId)
            return CreateEmbeddingsResponse(status="success", message="Embeddings created successfully from document.")
        except Exception as e:
            logger.error(f"Failed to create embeddings from document for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating embeddings from the document.")
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Source type '{request.sourceType.value}' not supported.")


@router.delete("/embeddings", response_model=DeleteEmbeddingsResponse)
async def delete_embeddings(
    request: DeleteEmbeddingsRequest,
):
    logger.info(f"Received delete embeddings request: {request.model_dump_json(indent=2)}")

    if request.sourceType == SourceType.WEB_PAGE:
        if not request.sourceData.webPageURL:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="webPageURL is required for WEB_PAGE source type")
        try:
            deleted_count = delete_data_from_website(request.sourceData.webPageURL, request.practiceId)
            return DeleteEmbeddingsResponse(
                status="success",
                message=f"Deletion successful for web page. {deleted_count} documents removed.",
                deleted_count=deleted_count,
            )
        except Exception as e:
            logger.error(f"Failed to delete embeddings from web page {request.sourceData.webPageURL} for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while deleting embeddings from the web page.")
    elif request.sourceType == SourceType.QA_PAIR:
        if not request.sourceData.qa_pair or not request.sourceData.qa_pair.question:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="qa_pair with question is required for QA_PAIR source type")
        try:
            deleted_count = delete_data_from_qa_pair(request.sourceData.qa_pair.question, request.practiceId)
            return DeleteEmbeddingsResponse(
                status="success",
                message=f"Deletion successful for Q&A pair. {deleted_count} documents removed.",
                deleted_count=deleted_count,
            )
        except Exception as e:
            logger.error(f"Failed to delete embeddings from Q&A pair for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while deleting embeddings from the Q&A pair.")
    elif request.sourceType == SourceType.DOCUMENT:
        if not request.sourceData.document or not request.sourceData.document.name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document with name is required for DOCUMENT source type")
        try:
            deleted_count = delete_data_from_document(request.sourceData.document.name, request.practiceId)
            return DeleteEmbeddingsResponse(
                status="success",
                message=f"Deletion successful for document. {deleted_count} documents removed.",
                deleted_count=deleted_count,
            )
        except Exception as e:
            logger.error(f"Failed to delete embeddings from document for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while deleting embeddings from the document.")
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Source type '{request.sourceType.value}' not supported.")
