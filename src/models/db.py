import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=True)
    status = Column(String, default="PENDING") # PENDING, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow)

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    status = Column(String, default="PENDING") # PENDING, PROCESSING, COMPLETED, FAILED
    trace_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
