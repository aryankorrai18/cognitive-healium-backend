import os
import logging
import tempfile
from typing import Optional, List

logger = logging.getLogger("healium")


class CacheProvider:
    def get(self, tenant_id: str, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, tenant_id: str, key: str, value: str, ttl: int = 86400):
        raise NotImplementedError


class LocalCacheProvider(CacheProvider):
    def __init__(self):
        self._cache = {}

    def get(self, tenant_id: str, key: str) -> Optional[str]:
        return self._cache.get(f"{tenant_id}:{key}")

    def set(self, tenant_id: str, key: str, value: str, ttl: int = 86400):
        self._cache[f"{tenant_id}:{key}"] = value


class RedisCacheProvider(CacheProvider):
    def __init__(self, redis_url: str):
        import redis
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def get(self, tenant_id: str, key: str) -> Optional[str]:
        return self.redis.get(f"cache:{tenant_id}:{key}")

    def set(self, tenant_id: str, key: str, value: str, ttl: int = 86400):
        self.redis.setex(f"cache:{tenant_id}:{key}", ttl, value)


class StorageProvider:
    def save_screenshot(self, tenant_id: str, locator, index: int) -> str:
        raise NotImplementedError


class LocalStorageProvider(StorageProvider):
    def __init__(self):
        os.makedirs("static/screenshots", exist_ok=True)

    def save_screenshot(self, tenant_id: str, locator, index: int) -> str:
        path = f"static/screenshots/{tenant_id}_heal_{index}.png"
        try:
            locator.screenshot(path=path)
            return f"http://localhost:8000/{path}"
        except Exception:
            return ""


class S3StorageProvider(StorageProvider):
    def __init__(self, bucket_name: str):
        import boto3
        self.s3 = boto3.client("s3")
        self.bucket = bucket_name

    def save_screenshot(self, tenant_id: str, locator, index: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                locator.screenshot(path=tmp.name)
                key = f"{tenant_id}/heal_{index}.png"
                self.s3.upload_file(tmp.name, self.bucket, key)
                return f"https://{self.bucket}.s3.amazonaws.com/{key}"
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                return ""
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class EventStoreProvider:
    def save_event(self, event_dict: dict):
        raise NotImplementedError

    def get_events(self, tenant_id: str, limit: int = 50) -> List[dict]:
        raise NotImplementedError


class LocalEventStoreProvider(EventStoreProvider):
    def __init__(self):
        self.events = []

    def save_event(self, event_dict: dict):
        self.events.append(event_dict)

    def get_events(self, tenant_id: str, limit: int = 50) -> List[dict]:
        return [e for e in self.events if e.get("tenant_id") == tenant_id][-limit:]


class PostgresEventStoreProvider(EventStoreProvider):
    def __init__(self, database_url: str):
        from psycopg2 import pool
        self.pool = pool.ThreadedConnectionPool(1, 20, database_url)
        self._create_table()

    def _create_table(self):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS healing_events (
                        id SERIAL PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        original_locator TEXT,
                        healed_locator TEXT,
                        intent TEXT,
                        action TEXT,
                        confidence FLOAT,
                        source TEXT,
                        status TEXT,
                        reasoning TEXT,
                        screenshot_url TEXT
                    )
                """)
                conn.commit()
        finally:
            self.pool.putconn(conn)

    def save_event(self, event_dict: dict):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO healing_events
                    (tenant_id, original_locator, healed_locator, intent, action,
                     confidence, source, status, reasoning, screenshot_url)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    event_dict.get("tenant_id", "default"),
                    event_dict.get("original_locator"),
                    event_dict.get("healed_locator"),
                    event_dict.get("intent"),
                    event_dict.get("action"),
                    event_dict.get("confidence"),
                    event_dict.get("source"),
                    event_dict.get("status"),
                    event_dict.get("reasoning"),
                    event_dict.get("screenshot_url"),
                ))
                conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_events(self, tenant_id: str, limit: int = 50) -> List[dict]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT original_locator, healed_locator, confidence,
                           source, status, reasoning
                    FROM healing_events
                    WHERE tenant_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (tenant_id, limit))
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)


def get_providers():
    redis_url    = os.getenv("REDIS_URL")
    s3_bucket    = os.getenv("S3_BUCKET")
    database_url = os.getenv("DATABASE_URL")

    cache       = RedisCacheProvider(redis_url)    if redis_url    else LocalCacheProvider()
    storage     = S3StorageProvider(s3_bucket)     if s3_bucket    else LocalStorageProvider()
    event_store = PostgresEventStoreProvider(database_url) if database_url else LocalEventStoreProvider()

    logger.info(
        f"Providers: Cache={type(cache).__name__} | "
        f"Storage={type(storage).__name__} | "
        f"EventStore={type(event_store).__name__}"
    )
    return cache, storage, event_store
