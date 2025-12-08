import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from speech_analyzer.analyzer import analyze_speech_return_dict

app = FastAPI(title="Speech Analyzer API")

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    segments: int = Query(5, ge=1, le=50, description="Number of segments to divide the audio into")
):
    """
    Upload an audio file and get speech analysis.
    
    - **file**: audio file to analyze (wav, mp3, etc.)
    - **segments**: number of segments to divide audio into for analysis
    """
    # Validate file type
    if file.content_type.split('/')[0] != "audio":
        raise HTTPException(status_code=400, detail="Upload an audio file")

    # Save uploaded file to a temporary path
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as tmp:
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Run speech analysis
    try:
        result = analyze_speech_return_dict(temp_path, segments=segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return JSONResponse(content=result)
