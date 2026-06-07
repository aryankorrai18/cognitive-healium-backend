import os
import logging
import chromadb
from typing import Optional, List
from healium.models import HealingEvent

logger = logging.getLogger("healium")


class HealiumMemory:
    def __init__(self, db_path: str = "data/", tenant_id: str = "default"):
        self.tenant_id = tenant_id

        chroma_host = os.getenv("CHROMA_HOST")
        if chroma_host:
            chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host, port=chroma_port
            )
            logger.info(f"ChromaDB: HTTP client -> {chroma_host}:{chroma_port}")
        else:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            logger.info(f"ChromaDB: local persistent client -> {db_path}")

        self.collection = self.chroma_client.get_or_create_collection(
            name=f"healing_{self.tenant_id}",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"ChromaDB collection: healing_{self.tenant_id} "
            f"({self.collection.count()} existing records)"
        )

    def query_vector_memory(self, intent_description: str,
                            n_results: int = 2) -> List[str]:
        count = self.collection.count()
        if count == 0:
            return []
        try:
            results = self.collection.query(
                query_texts=[intent_description],
                n_results=min(n_results, count)
            )
            if results and results["documents"]:
                return results["documents"][0]
            return []
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

    def save_to_vector_memory(self, event: HealingEvent):
        if event.status != "healed":
            return
        doc_text = (
            f"Intent: {event.intent}. "
            f"Old locator: {event.original_locator}. "
            f"New locator: {event.healed_locator}. "
            f"Reasoning: {event.reasoning}"
        )
        doc_id = (
            f"{self.tenant_id}_"
            f"{event.timestamp.replace(':', '-')}_"
            f"{event.original_locator.replace('#', '').replace('.', '')[:20]}"
        )
        try:
            self.collection.add(
                documents=[doc_text],
                ids=[doc_id],
                metadatas=[{
                    "tenant_id": self.tenant_id,
                    "old_locator": event.original_locator,
                    "new_locator": event.healed_locator,
                    "confidence": event.confidence,
                }]
            )
            logger.info(f"ChromaDB: saved healing pattern (tenant: {self.tenant_id})")
        except Exception as e:
            logger.error(f"ChromaDB save failed: {e}")
