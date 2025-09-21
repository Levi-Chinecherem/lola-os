# Standard imports
import typing as tp
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path
import uuid

# Third-party
try:
    from haystack import Pipeline
    from haystack.nodes import PromptNode, PromptTemplate
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.retrievers.in_memory import InMemoryDocumentRetriever
    from haystack.readers import TransformersSummarizer
    from haystack.utils import print_answers
except ImportError:
    raise ImportError("Haystack not installed. Run 'poetry add farm-haystack[all,inference]'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.rag.multimodal import MultiModalRetriever  # Phase 2 interconnection
from lola.libs.vector_dbs.adapter import get_vector_db_adapter  # VectorDB flexibility
from sentry_sdk import capture_exception

"""
File: Haystack pipeline adapter for LOLA OS RAG capabilities.
Purpose: Integrates Haystack's advanced NLP pipelines as alternative RAG engines 
         alongside LlamaIndex, with unified interface to LOLA's MultiModalRetriever.
How: Wraps Haystack Pipeline components (retrievers, readers, generators) 
     and maps to LOLA's RAG abstractions; supports config-based switching 
     between Haystack, LlamaIndex, and custom retrievers.
Why: Provides production-grade NLP capabilities (question answering, summarization, 
     dense passage retrieval) while maintaining LOLA's developer sovereignty 
     and VectorDB flexibility.
Full Path: lola-os/python/lola/libs/haystack/pipeline.py
"""

class HaystackPipelineAdapter:
    """HaystackPipelineAdapter: Wrapper for Haystack NLP pipelines in LOLA RAG.
    Does NOT create pipelines upfront—lazy initialization based on use case."""

    SUPPORTED_PIPELINES = {
        "retrieval": "document_retrieval",
        "qa": "question_answering", 
        "summarization": "text_summarization",
        "dense_retrieval": "dense_passage_retrieval"
    }

    def __init__(self):
        """
        Initializes Haystack adapter with LOLA configuration.
        Does Not: Load models—deferred until pipeline creation.
        """
        config = get_config()
        self.enabled = config.get("use_haystack", False)
        self.sentry_dsn = config.get("sentry_dsn", None)
        
        if not self.enabled:
            logger.warning("Haystack integration disabled in config")
            return

        # Haystack configuration
        self.model_provider = config.get("haystack_model_provider", "openai")
        self.embedding_model = config.get("haystack_embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.qa_model = config.get("haystack_qa_model", "distilbert-base-cased-distilled-squad")
        self.summarization_model = config.get("haystack_summarization_model", "facebook/bart-large-cnn")
        
        # VectorDB integration
        vector_config = config.get("vector_db", {})
        self.vector_db_adapter = self._initialize_vector_db(vector_config)
        
        # Document store (bridges Haystack and LOLA VectorDB)
        self.document_store = self._create_document_store()
        
        logger.info("Haystack pipeline adapter initialized")

    def _initialize_vector_db(self, config: Dict[str, Any]) -> tp.Any:
        """
        Initializes VectorDB adapter for Haystack document storage.
        Args:
            config: VectorDB configuration from LOLA config.
        Returns:
            VectorDB adapter instance.
        """
        try:
            adapter = get_vector_db_adapter(config)
            adapter.connect()
            logger.debug(f"Haystack VectorDB initialized: {config.get('type', 'unknown')}")
            return adapter
        except Exception as exc:
            logger.error(f"Haystack VectorDB initialization failed: {str(exc)}")
            if self.sentry_dsn:
                capture_exception(exc)
            # Fallback to in-memory for Haystack
            from haystack.document_stores import InMemoryDocumentStore
            return InMemoryDocumentStore()

    def _create_document_store(self) -> tp.Any:
        """
        Creates Haystack-compatible document store.
        Returns:
            Haystack DocumentStore instance.
        """
        if not self.enabled:
            from haystack.document_stores import InMemoryDocumentStore
            return InMemoryDocumentStore()

        # Bridge between LOLA VectorDB and Haystack
        class LolaHaystackBridge:
            """Bridge between LOLA VectorDB and Haystack document store interface."""
            
            def __init__(self, vector_db_adapter):
                self.adapter = vector_db_adapter
                self._documents: List[Dict] = []
            
            def write_documents(self, documents: List[Dict], **kwargs) -> None:
                """Haystack write_documents interface."""
                try:
                    embeddings = [doc.get("embedding", []) for doc in documents]
                    texts = [doc.get("content", "") for doc in documents]
                    metadatas = [doc.get("meta", {}) for doc in documents]
                    ids = [doc.get("id", str(uuid.uuid4())) for doc in documents]
                    
                    # Store in LOLA VectorDB
                    self.adapter.index(embeddings, texts, metadatas, ids)
                    
                    # Cache for Haystack compatibility
                    self._documents.extend(documents)
                    
                    logger.debug(f"Haystack bridge: wrote {len(documents)} documents")
                    
                except Exception as exc:
                    logger.error(f"Haystack document write failed: {str(exc)}")
                    if get_config().get("sentry_dsn"):
                        capture_exception(exc)
                    raise
            
            def get_document_count(self, **kwargs) -> int:
                """Returns document count."""
                return self.adapter.get_stats().get("count", 0)
            
            def get_all_documents(self, **kwargs) -> List[Dict]:
                """Returns all documents (limited for performance)."""
                return self._documents[:1000]  # Limit for memory
            
            def delete(self, ids: List[str]) -> None:
                """Deletes documents by ID."""
                self.adapter.delete(ids)
                self._documents = [d for d in self._documents if d.get("id") not in ids]

        return LolaHaystackBridge(self.vector_db_adapter)

    def create_retrieval_pipeline(self, top_k: int = 5, 
                                filters: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Creates Haystack retrieval pipeline.
        Args:
            top_k: Number of documents to retrieve.
            filters: Metadata filters.
        Returns:
            Haystack Pipeline instance.
        """
        if not self.enabled:
            raise ValueError("Haystack disabled in configuration")

        try:
            # Create retriever
            retriever = InMemoryDocumentRetriever(
                document_store=self.document_store,
                embedding_model=self.embedding_model,
                scale_score=False
            )
            
            # Create pipeline
            pipeline = Pipeline()
            pipeline.add_node(
                component=retriever,
                name="retriever",
                inputs=["Query"]
            )
            
            logger.debug(f"Haystack retrieval pipeline created (top_k={top_k})")
            return pipeline
            
        except Exception as exc:
            self._handle_error(exc, "retrieval pipeline creation")
            raise

    def create_qa_pipeline(self, top_k_retriever: int = 5, 
                          top_k_reader: int = 1,
                          reader_model: Optional[str] = None) -> Pipeline:
        """
        Creates question answering pipeline.
        Args:
            top_k_retriever: Documents to retrieve.
            top_k_reader: Answers to extract per document.
            reader_model: Optional custom reader model.
        Returns:
            QA Pipeline instance.
        """
        if not self.enabled:
            raise ValueError("Haystack disabled in configuration")

        try:
            # Use configured model or default
            qa_model = reader_model or self.qa_model
            
            # Create retriever
            retriever = InMemoryDocumentRetriever(
                document_store=self.document_store,
                embedding_model=self.embedding_model,
                scale_score=False
            )
            
            # Create reader (extractive QA)
            from haystack.nodes import FARMReader
            reader = FARMReader(
                model_name_or_path=qa_model,
                use_gpu=True,
                return_on_no_answer=False
            )
            
            # Create pipeline
            p = Pipeline()
            p.add_node(component=retriever, name="retriever", inputs=["Query"])
            p.add_node(component=reader, name="reader", inputs=["retriever"])
            
            logger.info(f"Haystack QA pipeline created: {qa_model}")
            return p
            
        except Exception as exc:
            self._handle_error(exc, "QA pipeline creation")
            raise

    def create_summarization_pipeline(self, model_name: Optional[str] = None) -> Pipeline:
        """
        Creates text summarization pipeline.
        Args:
            model_name: Optional custom summarization model.
        Returns:
            Summarization Pipeline instance.
        """
        if not self.enabled:
            raise ValueError("Haystack disabled in configuration")

        try:
            # Use configured model or default
            model = model_name or self.summarization_model
            
            # Create summarizer
            summarizer = TransformersSummarizer(
                model_name_or_path=model,
                use_gpu=True
            )
            
            # Create simple pipeline
            pipeline = Pipeline()
            pipeline.add_node(
                component=summarizer,
                name="summarizer",
                inputs=["Query"]
            )
            
            logger.debug(f"Haystack summarization pipeline created: {model}")
            return pipeline
            
        except Exception as exc:
            self._handle_error(exc, "summarization pipeline creation")
            raise

    def run_retrieval(self, pipeline: Pipeline, query: str, 
                     top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Runs retrieval pipeline and formats results for LOLA.
        Args:
            pipeline: Haystack retrieval pipeline.
            query: Search query.
            top_k: Number of results.
            filters: Metadata filters.
        Returns:
            LOLA-formatted results list.
        """
        if not self.enabled:
            return []

        try:
            # Run pipeline
            result = pipeline.run(
                query=query,
                params={"Retriever": {"top_k": top_k}},
                debug=True
            )
            
            # Extract and format documents
            documents = result.get("answers", [])
            
            formatted_results = []
            for doc in documents[:top_k]:
                if hasattr(doc, 'document') and doc.document:
                    haystack_doc = doc.document
                    result = {
                        "id": haystack_doc.id or str(uuid.uuid4()),
                        "text": haystack_doc.content,
                        "score": getattr(doc, 'score', 0.0),
                        "metadata": getattr(haystack_doc, 'meta', {})
                    }
                    
                    # Extract embedding if available
                    if hasattr(haystack_doc, 'embedding') and haystack_doc.embedding:
                        result["embedding"] = haystack_doc.embedding
                    
                    formatted_results.append(result)
            
            logger.debug(f"Haystack retrieval: {len(formatted_results)} results for '{query[:50]}...'")
            return formatted_results
            
        except Exception as exc:
            self._handle_error(exc, f"Haystack retrieval: {query[:50]}...")
            return []

    def run_qa(self, pipeline: Pipeline, question: str, 
              top_k: int = 1) -> Dict[str, Any]:
        """
        Runs QA pipeline.
        Args:
            pipeline: Haystack QA pipeline.
            question: Question to answer.
            top_k: Number of answers.
        Returns:
            Answer with context and confidence.
        """
        if not self.enabled:
            return {"answer": "", "confidence": 0.0, "context": ""}

        try:
            result = pipeline.run(
                query=question,
                params={"Retriever": {"top_k": 10}, "Reader": {"top_k": top_k}},
                debug=True
            )
            
            if result.get("answers"):
                top_answer = result["answers"][0]
                if hasattr(top_answer, 'answer') and top_answer.answer:
                    return {
                        "answer": top_answer.answer,
                        "confidence": getattr(top_answer, 'score', 0.0),
                        "context": top_answer.document.content if hasattr(top_answer, 'document') else "",
                        "sources": [top_answer.document.meta if hasattr(top_answer, 'document') else {}]
                    }
            
            logger.debug(f"Haystack QA: no answer found for '{question[:50]}...'")
            return {"answer": "", "confidence": 0.0, "context": ""}
            
        except Exception as exc:
            self._handle_error(exc, f"Haystack QA: {question[:50]}...")
            return {"answer": "", "confidence": 0.0, "context": ""}

    def integrate_with_lola_rag(self, rag_component: MultiModalRetriever) -> None:
        """
        Registers Haystack pipelines with LOLA RAG system.
        Args:
            rag_component: LOLA MultiModalRetriever instance.
        """
        if not self.enabled:
            logger.warning("Cannot integrate Haystack - adapter disabled")
            return

        try:
            # Create retrieval pipeline for integration
            retrieval_pipeline = self.create_retrieval_pipeline(top_k=5)
            
            def haystack_retrieve(query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
                """Haystack retriever callback for LOLA RAG."""
                return self.run_retrieval(retrieval_pipeline, query, top_k)
            
            # Register with RAG component
            rag_component.register_retriever("haystack", haystack_retrieve)
            logger.info("Haystack retriever registered with LOLA RAG")
            
        except Exception as exc:
            self._handle_error(exc, "Haystack RAG integration")
            logger.error(f"Failed to integrate Haystack with LOLA RAG: {str(exc)}")

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Centralized error handling for Haystack operations.
        Args:
            exc: Exception to handle.
            context: Context string.
        """
        logger.error(f"Haystack {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Adds documents to Haystack document store.
        Args:
            documents: List of document dictionaries with content and metadata.
        """
        if not self.enabled:
            return

        try:
            # Convert LOLA documents to Haystack format
            haystack_docs = []
            for doc in documents:
                haystack_doc = {
                    "content": doc.get("text", ""),
                    "meta": doc.get("metadata", {}),
                    "id": doc.get("id", str(uuid.uuid4()))
                }
                # Add embedding if available
                if "embedding" in doc:
                    haystack_doc["embedding"] = doc["embedding"]
                haystack_docs.append(haystack_doc)
            
            # Write to document store
            self.document_store.write_documents(haystack_docs)
            logger.info(f"Added {len(haystack_docs)} documents to Haystack store")
            
        except Exception as exc:
            self._handle_error(exc, "document addition")
            raise


# Convenience functions
def create_haystack_retrieval(documents: List[Dict[str, Any]] = None, 
                            top_k: int = 5) -> Pipeline:
    """Quick function to create and populate retrieval pipeline."""
    adapter = HaystackPipelineAdapter()
    pipeline = adapter.create_retrieval_pipeline(top_k)
    
    if documents:
        adapter.add_documents(documents)
    
    return pipeline

def run_haystack_qa(question: str, documents: List[Dict[str, Any]], 
                   top_k: int = 1) -> Dict[str, Any]:
    """Quick QA function with document context."""
    adapter = HaystackPipelineAdapter()
    if documents:
        adapter.add_documents(documents)
    
    pipeline = adapter.create_qa_pipeline(top_k_retriever=10)
    return adapter.run_qa(pipeline, question, top_k)

__all__ = [
    "HaystackPipelineAdapter",
    "create_haystack_retrieval",
    "run_haystack_qa"
]