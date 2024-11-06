from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx
from services.streaming import detection_stream

# Initialize FastAPI app
app = FastAPI()

# Define FastAPI endpoint
@app.get("/stream_detections")
async def stream_detections():
    # Use an async HTTP client to stream data to the external URL
    async with httpx.AsyncClient() as client:
        async for data in detection_stream():
            # Send each chunk of data as a stream
            await client.post(
                "https://api.forceai.tech/crime-detection/send-event",
                content=data,
                headers={"Content-Type": "application/json"},
            )
    return {"status": "Streaming to external URL"}
