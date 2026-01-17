from fastapi import FastAPI
from backend.debris_api import router as debris_router

app = FastAPI(title="Space Debris Recycling Backend")

app.include_router(debris_router, prefix="/debris")

@app.get("/")
def health():
    return {"status": "Backend running"}