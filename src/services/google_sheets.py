import logging
from typing import List, Optional

import gspread
from google.oauth2.service_account import Credentials

from src.config import settings

logger = logging.getLogger(__name__)


class GoogleSheetsService:
    """
    A service to interact with the Google Sheets API.
    """

    def __init__(self):
        self.creds = self._authenticate()
        self.client = gspread.authorize(self.creds)

    def _authenticate(self) -> Credentials:
        """
        Authenticates with Google Sheets using service account credentials
        from environment variables.

        Returns:
            The authenticated credentials object.
        """
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ]

            # Handle escaped newlines in the private key from environment variables
            private_key = settings.GOOGLE_SA_PRIVATE_KEY.replace("\\n", "\n")

            service_account_info = {
                "type": settings.GOOGLE_SA_TYPE,
                "project_id": settings.GOOGLE_SA_PROJECT_ID,
                "private_key_id": settings.GOOGLE_SA_PRIVATE_KEY_ID,
                "private_key": private_key,
                "client_email": settings.GOOGLE_SA_CLIENT_EMAIL,
                "client_id": settings.GOOGLE_SA_CLIENT_ID,
                "auth_uri": settings.GOOGLE_SA_AUTH_URI,
                "token_uri": settings.GOOGLE_SA_TOKEN_URI,
                "auth_provider_x509_cert_url": settings.GOOGLE_SA_AUTH_PROVIDER_X509_CERT_URL,
                "client_x509_cert_url": settings.GOOGLE_SA_CLIENT_X509_CERT_URL,
            }

            creds = Credentials.from_service_account_info(
                service_account_info, scopes=scopes
            )
            logger.debug("Successfully authenticated with Google Sheets.")
            return creds
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets: {e}", exc_info=True)
            raise

    def get_worksheet(
        self, spreadsheet_id: str, worksheet_name: str
    ) -> Optional[gspread.Worksheet]:
        """
        Gets a specific worksheet from a spreadsheet.

        Args:
            spreadsheet_id: The ID of the Google Spreadsheet.
            worksheet_name: The name of the worksheet.

        Returns:
            A gspread.Worksheet object or None if not found.
        """
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(worksheet_name)
            return worksheet
        except gspread.exceptions.SpreadsheetNotFound:
            logger.error(f"Spreadsheet with ID '{spreadsheet_id}' not found.")
            return None
        except gspread.exceptions.WorksheetNotFound:
            logger.error(
                f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet_id}'."
            )
            return None
        except Exception as e:
            logger.error(
                f"An error occurred while accessing spreadsheet '{spreadsheet_id}': {e}"
            )
            return None

    def read_data(self, worksheet: gspread.Worksheet) -> List[dict]:
        """
        Reads all data from a worksheet as a list of dictionaries.

        Args:
            worksheet: The gspread.Worksheet object to read from.

        Returns:
            A list of dictionaries representing the rows.
        """
        try:
            return worksheet.get_all_records()
        except Exception as e:
            logger.error(f"Failed to read data from worksheet: {e}")
            raise

    def write_data(self, worksheet: gspread.Worksheet, data: List[List[str]]):
        """
        Writes data to a worksheet. Note: This will overwrite existing data.

        Args:
            worksheet: The gspread.Worksheet object to write to.
            data: A list of lists representing the rows to write.
        """
        try:
            worksheet.update(data)
            logger.info(f"Successfully wrote {len(data)} rows to worksheet.")
        except Exception as e:
            logger.error(f"Failed to write data to worksheet: {e}")
            raise

    def append_row(self, worksheet: gspread.Worksheet, row: List[str]):
        """
        Appends a single row to a worksheet.

        Args:
            worksheet: The gspread.Worksheet object to append to.
            row: A list of values for the new row.
        """
        try:
            worksheet.append_row(row)
            logger.info("Successfully appended row to worksheet.")
        except Exception as e:
            logger.error(f"Failed to append row to worksheet: {e}")
            raise
