from sqlalchemy import Column, String, JSON, Boolean
from sqlalchemy.dialects.postgresql import JSONB

from .db import Base


class Interaction(Base):
    """
    Represents a conversation session stored in the database.
    """

    __tablename__ = "interactions"

    session_id = Column(String, primary_key=True, index=True)
    practice_id = Column(String, index=True, nullable=True)
    messages = Column(JSONB, nullable=False)
    states = Column(JSONB, nullable=False, server_default='["IDLE"]')
    interaction_data = Column(JSON, nullable=True)
