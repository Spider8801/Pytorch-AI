from __future__ import annotations

import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """
    All tunable knobs in one validated object.

    Pydantic validates values on construction, so bad settings raise
    immediately rather than at query time.
    """

    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace model id used for embedding chunks and queries.",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Target number of characters per chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Characters shared between adjacent chunks to avoid boundary loss.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of chunks retrieved from FAISS per query.",
    )
    generation_model: str = Field(
        default="claude-opus-4-5",
        description="Anthropic model used for answer generation.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, v: int, info) -> int:
        """Ensures overlap < chunk_size to avoid infinite splitting loops."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})."
            )
        return v


class QueryResult(BaseModel):
    """
    Structured output returned by RAGPipeline.query().

    Carries the answer, the source documents used, and latency telemetry.
    """

    query: str = Field(..., description="The original user question.")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved context.")
    source_documents: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of source labels whose chunks informed the answer.",
    )
    retrieval_time_ms: float = Field(default=0.0, description="FAISS retrieval time in ms.")
    generation_time_ms: float = Field(default=0.0, description="LLM generation time in ms.")

    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT = ChatPromptTemplate.from_template(
    """You are a precise question-answering assistant.
Answer the question using ONLY the information in the provided context.
If the context does not contain enough information, say so explicitly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    LangChain-backed RAG pipeline.

    LangChain handles:
      - text splitting (RecursiveCharacterTextSplitter)
      - embedding          (HuggingFaceEmbeddings)
      - vector storage     (FAISS)
      - retrieval          (vectorstore.as_retriever)
      - prompt + LLM call  (LCEL chain via | operator)

    This class owns configuration, the vectorstore, and result shaping.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        # LangChain splitter — tries to break on paragraphs, then sentences,
        # then words before falling back to characters.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        # Local HuggingFace embedding model — no API key required.
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name
        )

        self._vectorstore: FAISS | None = None

        # LCEL chain: format prompt → call LLM → parse text out of the response.
        self._llm = ChatAnthropic(model=self.config.generation_model, max_tokens=1024)
        self._chain = PROMPT | self._llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, text: str, source: str = "document") -> int:
        """
        Split *text* into chunks, embed them, and add them to the FAISS index.

        LangChain Documents carry arbitrary metadata; we store `source` so
        retrieval results can be attributed to the originating document.

        Parameters
        ----------
        text   : Raw document content.
        source : Human-readable label (filename, URL, …).

        Returns
        -------
        int  Number of chunks added to the index.
        """
        if not text or not text.strip():
            raise ValueError("Cannot ingest empty text.")

        # Wrap raw text in LangChain Document so metadata travels with the chunk.
        docs: list[Document] = self._splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source}],
        )

        if self._vectorstore is None:
            # First ingestion: create the FAISS index from scratch.
            self._vectorstore = FAISS.from_documents(docs, self._embeddings)
        else:
            # Subsequent ingestions: add to the existing index.
            self._vectorstore.add_documents(docs)

        return len(docs)

    def query(self, question: str) -> QueryResult:
        """
        Retrieve the most relevant chunks and generate a grounded answer.

        The LCEL chain ( PROMPT | LLM | parser ) is invoked with a dict
        that contains both the formatted context string and the raw question.

        Parameters
        ----------
        question : Natural-language question.

        Returns
        -------
        QueryResult  Pydantic model with answer, sources, and latency.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        if self._vectorstore is None:
            raise RuntimeError("Index is empty — call ingest() first.")

        # --- Retrieval ---
        retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        t0 = time.perf_counter()
        retrieved_docs: list[Document] = retriever.invoke(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Collapse retrieved chunks into a single context string for the prompt.
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in retrieved_docs
        )

        # --- Generation ---
        t1 = time.perf_counter()
        answer: str = self._chain.invoke({"context": context, "question": question})
        generation_ms = (time.perf_counter() - t1) * 1000

        # Deduplicate source labels while preserving order.
        seen: set[str] = set()
        sources: list[str] = []
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        return QueryResult(
            query=question,
            answer=answer,
            source_documents=sources,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
        )

    @property
    def index_size(self) -> int:
        """Number of chunks currently in the FAISS index."""
        return self._vectorstore.index.ntotal if self._vectorstore else 0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    docs = {
        "python_intro": (
            "Python is a high-level, interpreted programming language known for its clear "
            "syntax and readability. Created by Guido van Rossum and first released in 1991, "
            "Python supports procedural, object-oriented, and functional programming paradigms."
        ),
        "machine_learning": (
            "Machine learning is a subset of artificial intelligence that gives computers "
            "the ability to learn from data without being explicitly programmed. Key categories "
            "include supervised, unsupervised, and reinforcement learning."
        ),
        "rag_overview": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative "
            "language model. Relevant documents are fetched from a vector store and injected into "
            "the prompt, grounding the model's output and reducing hallucination."
        ),
    }

    pipeline = RAGPipeline(PipelineConfig(chunk_size=300, chunk_overlap=50, top_k=3))

    print("Ingesting …")
    for name, text in docs.items():
        n = pipeline.ingest(text, source=name)
        print(f"  {name}: {n} chunk(s)  (index size: {pipeline.index_size})")

    questions = [
        "What is Python and who created it?",
        "How does RAG reduce hallucination?",
    ]

    print("\nQuerying …\n" + "=" * 60)
    for q in questions:
        r = pipeline.query(q)
        print(f"\nQ: {r.query}")
        print(f"A: {r.answer}")
        print(f"   Sources: {r.source_documents} | Total: {r.total_time_ms:.0f} ms")
        print("-" * 60)


if __name__ == "__main__":
    main()