# Standard imports
import typing as tp
from typing import List, Dict, Any, Optional
import json
from contextlib import contextmanager
import uuid

# Third-party
try:
    from sqlalchemy import create_engine, text, Column, Integer, String, LargeBinary
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    import numpy as np
    import psycopg2
    from pgvector.sqlalchemy import Vector
except ImportError as e:
    raise ImportError(f"Postgres/pgvector dependencies missing: {e}. "
                     "Run 'poetry add psycopg2-binary sqlalchemy pgvector'")

# Local
from lola.libs.vector_dbs.adapter import VectorDBAdapter
from lola.utils.config import get_config
from lola.utils.logging import logger
from sentry_sdk import capture_exception

"""
File: PostgreSQL/pgvector VectorDB implementation for LOLA OS.
Purpose: Provides relational vector storage using PostgreSQL with pgvector extension 
         for hybrid SQL + vector search capabilities.
How: Uses SQLAlchemy ORM with pgvector Vector type; supports full-text search 
     alongside vector similarity; handles connection pooling and transactions.
Why: Perfect for production deployments requiring ACID transactions, complex 
     metadata queries, and integration with existing Postgres infrastructure.
Full Path: lola-os/python/lola/libs/vector_dbs/postgres.py
"""

Base = declarative_base()

class VectorDocument(Base):
    """SQLAlchemy model for vector documents."""
    __tablename__ = 'lola_vectors'
    
    id = Column(String, primary_key=True)
    embedding = Column(Vector(1536))  # Default dimension, configurable
    text = Column(String)  # Full text content
    text_length = Column(Integer)
    metadata = Column(String)  # JSON metadata
    created_at = Column(String)
    updated_at = Column(String)

class PostgresVectorDBAdapter(VectorDBAdapter):
    """PostgresVectorDBAdapter: Relational vector storage with pgvector.
    Does NOT create database—assumes existing Postgres with pgvector extension."""

    DEFAULT_TABLE_NAME = "lola_vectors"
    DEFAULT_DSN = "postgresql://user:password@localhost:5432/lola_db"

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes Postgres adapter.
        Args:
            config: Configuration with 'dsn' and 'table_name'.
        """
        super().__init__(config)
        self.dsn = config.get("dsn", self.DEFAULT_DSN)
        self.table_name = config.get("table_name", self.DEFAULT_TABLE_NAME)
        self.embedding_dim = config.get("embedding_dim", 1536)
        self.pool_size = config.get("pool_size", 5)
        self.max_overflow = config.get("max_overflow", 10)
        self.connect_timeout = config.get("connect_timeout", 10)
        
        # Update model with actual dimension
        VectorDocument.embedding = Column(Vector(self.embedding_dim))
        VectorDocument.__table__.name = self.table_name
        
        self.engine = None
        self.Session = None
        self._connected = False
        
        logger.info(f"Postgres adapter initialized: {self.dsn} (table: {self.table_name})")

    def connect(self) -> None:
        """Connects to PostgreSQL and creates table if needed."""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.dsn,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=20,
                connect_args={
                    "connect_timeout": self.connect_timeout,
                    "options": "-c search_path=lola_vectors"  # Use custom schema if needed
                }
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug("Postgres connection successful")
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Ensure pgvector extension
            self._ensure_pgvector_extension()
            
            # Create table
            VectorDocument.metadata.create_all(self.engine)
            
            self._connected = True
            stats = self.get_stats()
            logger.info(f"Postgres connected: {stats['count']} vectors in {self.embedding_dim}D")
            
        except Exception as exc:
            self._handle_error(exc, "Postgres connection")
            raise

    def _ensure_pgvector_extension(self) -> None:
        """Ensures pgvector extension is enabled."""
        try:
            with self.engine.connect() as conn:
                # Check if extension exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension 
                        WHERE extname = 'vector'
                    );
                """)).scalar()
                
                if not result:
                    # Create extension
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    logger.info("Created pgvector extension")
                else:
                    logger.debug("pgvector extension already exists")
                    
        except Exception as exc:
            logger.warning(f"pgvector extension check failed: {str(exc)}")
            # Continue without extension (will fail on vector operations)

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        if not self._connected:
            raise RuntimeError("Postgres adapter not connected")
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def disconnect(self) -> None:
        """Disposes engine and closes connections."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.debug("Postgres engine disposed")
            
            self.engine = None
            self.Session = None
            self._connected = False
            
        except Exception as exc:
            logger.warning(f"Postgres disconnect warning: {str(exc)}")

    def index(self, embeddings: List[List[float]], texts: List[str], 
              metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """Indexes embeddings in PostgreSQL table."""
        self.ensure_connected()
        
        try:
            n_vectors = len(embeddings)
            if n_vectors != len(texts) or n_vectors != len(metadatas):
                raise ValueError("All input lists must have equal lengths")
            
            if any(len(emb) != self.embedding_dim for emb in embeddings):
                raise ValueError(f"All embeddings must have {self.embedding_dim} dimensions")

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(n_vectors)]

            with self.get_session() as session:
                # Prepare documents
                documents = []
                for i, (embedding, text, metadata, doc_id) in enumerate(
                    zip(embeddings, texts, metadatas, ids)
                ):
                    doc = VectorDocument(
                        id=doc_id,
                        embedding=embedding,
                        text=text,
                        text_length=len(text),
                        metadata=json.dumps(metadata),
                        created_at=str(tp.datetime.now().isoformat()),
                        updated_at=str(tp.datetime.now().isoformat())
                    )
                    documents.append(doc)
                
                # Bulk insert
                session.add_all(documents)
                
                # Verify insertion
                inserted_count = session.query(VectorDocument).filter(
                    VectorDocument.id.in_(ids)
                ).count()
                
                logger.info(f"Indexed {inserted_count} vectors in Postgres")
                
        except Exception as exc:
            self._handle_error(exc, "Postgres indexing")
            raise

    def query(self, embedding: List[float], top_k: int = 5, 
              include_metadata: bool = True,
              where: Optional[Dict[str, Any]] = None,
              text_contains: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries PostgreSQL with vector similarity and optional filters.
        Args:
            embedding: Query embedding.
            top_k: Number of results.
            include_metadata: Include metadata.
            where: Metadata filter (JSON conditions).
            text_contains: Full-text search term.
        Returns:
            List of results with similarity scores.
        """
        self.ensure_connected()
        
        try:
            with self.get_session() as session:
                # Build query
                query = session.query(VectorDocument)
                
                # Vector similarity search using cosine distance
                similarity_scores = []
                for i, emb in enumerate(session.query(VectorDocument.embedding).all()):
                    # Compute cosine similarity
                    if emb[0] is not None:
                        sim = np.dot(np.array(embedding), np.array(emb[0])) / (
                            np.linalg.norm(np.array(embedding)) * np.linalg.norm(np.array(emb[0]))
                        )
                        similarity_scores.append((i, sim))
                
                # Sort by similarity (descending)
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in similarity_scores[:top_k]]
                
                # Apply filters
                if where:
                    # Convert JSON metadata filter to SQL
                    filter_conditions = []
                    for key, value in where.items():
                        filter_conditions.append(
                            VectorDocument.metadata.contains(json.dumps({key: value}))
                        )
                    query = query.filter(*filter_conditions)
                
                if text_contains:
                    # Full-text search
                    query = query.filter(
                        VectorDocument.text.contains(text_contains)
                    )
                
                # Get results
                results = query.filter(
                    VectorDocument.id.in_(
                        [session.query(VectorDocument.id).offset(idx).limit(1).scalar() 
                         for idx in top_indices]
                    )
                ).all()
                
                formatted_results = []
                for doc in results:
                    if doc.embedding is not None:
                        # Recalculate similarity for this document
                        sim = np.dot(np.array(embedding), np.array(doc.embedding)) / (
                            np.linalg.norm(np.array(embedding)) * np.linalg.norm(np.array(doc.embedding))
                        )
                        
                        result = {
                            "id": doc.id,
                            "distance": float(1.0 - sim),  # Convert to distance
                            "text": doc.text,
                            "text_length": doc.text_length
                        }
                        
                        if include_metadata and doc.metadata:
                            try:
                                result["metadata"] = json.loads(doc.metadata)
                            except json.JSONDecodeError:
                                result["metadata"] = {}
                        
                        formatted_results.append(result)
                
                return formatted_results[:top_k]
            
        except Exception as exc:
            self._handle_error(exc, "Postgres query")
            raise

    def delete(self, ids: List[str]) -> None:
        """Deletes vectors by IDs."""
        self.ensure_connected()
        
        try:
            with self.get_session() as session:
                before_count = session.query(VectorDocument).count()
                
                result = session.query(VectorDocument).filter(
                    VectorDocument.id.in_(ids)
                ).delete(synchronize_session=False)
                
                session.commit()
                
                after_count = session.query(VectorDocument).count()
                deleted_count = before_count - after_count
                
                logger.info(f"Postgres deleted {result} vectors (remaining: {after_count})")
                
        except Exception as exc:
            self._handle_error(exc, "Postgres delete")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Returns PostgreSQL table statistics."""
        self.ensure_connected()
        try:
            with self.get_session() as session:
                count = session.query(VectorDocument).count()
                
                # Get dimension from first row or use config
                dim_result = session.query(VectorDocument.embedding).first()
                actual_dim = len(dim_result.embedding) if dim_result and dim_result.embedding else self.embedding_dim
                
                return {
                    "type": "postgres",
                    "count": count,
                    "dimensions": actual_dim,
                    "table_name": self.table_name,
                    "database_connections": self.engine.pool.size()
                }
        except Exception as exc:
            self._handle_error(exc, "Postgres stats")
            return {"type": "postgres", "count": 0, "dimensions": self.embedding_dim}

    def _handle_error(self, exc: Exception, context: str) -> None:
        """Error handling for Postgres operations."""
        full_context = f"Postgres[{self.table_name}] {context}"
        logger.error(f"{full_context}: {str(exc)}")
        config = get_config()
        if config.get("sentry_dsn"):
            capture_exception(exc)


# Factory function
def create_postgres_adapter(
    dsn: str = "postgresql://user:password@localhost:5432/lola_db",
    table_name: str = "lola_vectors",
    embedding_dim: int = 1536,
    pool_size: int = 5
) -> PostgresVectorDBAdapter:
    """Creates Postgres adapter with defaults."""
    config = {
        "type": "postgres",
        "dsn": dsn,
        "table_name": table_name,
        "embedding_dim": embedding_dim,
        "pool_size": pool_size
    }
    return PostgresVectorDBAdapter(config)

__all__ = [
    "PostgresVectorDBAdapter",
    "VectorDocument",
    "create_postgres_adapter"
]