"""
Entry point for Railway deployment.
This file exists at the backend root so Railway can auto-detect and run it.
"""
import uvicorn
from app.main import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

