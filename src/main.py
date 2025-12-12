import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.chatflow.router import router as chatflow_router
from src.api.embeddings.router import router as embeddings_router
from src.config import settings
from src.database.db import engine, test_db_connection
from src.services.google_sheets import GoogleSheetsService
from src.shared.schemas import HealthResponse

log_level = settings.LOG_LEVEL.upper()
logging.basicConfig(
    level=log_level,
    format="%(levelname)s:%(name)s: [%(funcName)s] - %(message)s",
)

# httpx logs at INFO level for requests, which is noisy for production.
# We set it to WARNING to silence it, unless we are in DEBUG mode.
if log_level != "DEBUG":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.debug("Starting up application...")
    if not await test_db_connection():
        logger.warning(
            "Database connection could not be established on startup."
        )
    else:
        logger.debug("Database connection successful.")

    try:
        app.state.sheets_service = GoogleSheetsService()
        logger.info("Google Sheets Service initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Google Sheets Service: {e}")
        app.state.sheets_service = None

    yield
    # Shutdown
    logger.info("Shutting down application...")
    await engine.dispose()

    yield
    # Shutdown
    logger.info("Shutting down application...")
    await engine.dispose()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unhandled exception occurred for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An internal server error occurred. Please check the logs for details."},
    )


app.include_router(chatflow_router, prefix="/api/v1", tags=["Chatflow"])
app.include_router(embeddings_router, prefix="/api/v1", tags=["Embeddings"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """
    Checks the health of the application and its database connection.
    """
    db_ok = await test_db_connection()
    sheets_ok = request.app.state.sheets_service is not None

    return HealthResponse(
        status="ok",
        db_connection="ok" if db_ok else "failed",
        sheets_connection="ok" if sheets_ok else "failed",
    )
