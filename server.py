import os
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Cognitive Healium Dashboard")

os.makedirs("static/screenshots", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

if not os.getenv("S3_BUCKET"):
    app.mount(
        "/static/screenshots",
        StaticFiles(directory="static/screenshots"),
        name="screenshots",
    )

connected_clients: list[WebSocket] = []

redis_url    = os.getenv("REDIS_URL")
redis_client = None

if redis_url:
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url(redis_url, decode_responses=True)


@app.on_event("startup")
async def startup_event():
    if redis_client:
        asyncio.create_task(redis_listener())


async def redis_listener():
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("heal_events")
    async for message in pubsub.listen():
        if message["type"] == "message":
            data = message["data"]
            for client in connected_clients.copy():
                try:
                    await client.send_text(data)
                except Exception:
                    connected_clients.remove(client)


@app.get("/")
async def get_dashboard():
    with open("static/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        connected_clients.remove(websocket)


@app.post("/api/heal")
async def receive_healing_event(event: dict):
    if redis_client:
        await redis_client.publish("heal_events", json.dumps(event))
    else:
        for client in connected_clients.copy():
            try:
                await client.send_text(json.dumps(event))
            except Exception:
                connected_clients.remove(client)
    return {"status": "received"}
