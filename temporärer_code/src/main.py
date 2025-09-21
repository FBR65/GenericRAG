from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import io
import zipfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
import uuid
import logging
from datetime import datetime

from .extractor import PDFExtractor
from .models import ExtractionResult
from .config import settings
from .utils import (
    setup_logging,
    validate_file_path,
    get_file_size,
    format_file_size,
    ensure_directory_exists,
)

# Logging einrichten
logger = setup_logging(settings.log_level, settings.log_file)

app = FastAPI(title="PDFExtractor API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stelle sicher, dass die Verzeichnisse existieren
ensure_directory_exists(settings.output_dir)
ensure_directory_exists(settings.upload_dir)
ensure_directory_exists(settings.temp_dir)


class ExtractionSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.status = "started"
        self.progress = 0
        self.result = None
        self.error = None


# In-memory Speicherung für Sessions (in Produktion: Redis/Datenbank)
sessions: Dict[str, ExtractionSession] = {}


@app.post("/extract")
async def extract_pdf(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    """Startet die PDF-Extraktion und gibt SSE-Stream zurück"""
    # Validiere Datei
    if not validate_file_path(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=400, detail="Nur PDF-Dateien werden unterstützt"
        )

    # Validiere Dateigröße
    file_size = get_file_size(file.filename) if hasattr(file, "size") else 0
    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"Datei zu groß. Maximal {format_file_size(settings.max_file_size)} erlaubt.",
        )

    session_id = str(uuid.uuid4())
    sessions[session_id] = ExtractionSession(session_id)

    # Speichere die Datei temporär
    temp_file_path = settings.temp_dir / f"temp_{session_id}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Datei: {e}")
        raise HTTPException(status_code=500, detail="Fehler beim Speichern der Datei")

    # Starte Background Task
    if background_tasks:
        background_tasks.add_task(
            process_pdf_extraction, str(temp_file_path), session_id
        )

    return {
        "session_id": session_id,
        "message": "Extraktion gestartet",
        "file_info": {"name": file.filename, "size": format_file_size(file_size)},
    }


@app.get("/extract/{session_id}/stream")
async def stream_extraction_progress(session_id: str):
    """SSE-Stream für Fortschrittsupdates"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session = sessions[session_id]

    async def event_generator() -> AsyncGenerator[str, None]:
        while session.status not in ["completed", "error"]:
            if session.status == "error":
                yield f"data: {json.dumps({'status': 'error', 'error': session.error})}\n\n"
                break

            yield f"data: {json.dumps({'status': session.status, 'progress': session.progress})}\n\n"
            await asyncio.sleep(1)

        # Sende abschließendes Update
        if session.status == "completed" and session.result:
            yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'result': session.result})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/extract/{session_id}/download")
async def download_result(session_id: str):
    """Lädt das Ergebnis als ZIP-Datei herunter"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session = sessions[session_id]
    if not session.result or session.status != "completed":
        raise HTTPException(
            status_code=400, detail="Extraktion noch nicht abgeschlossen"
        )

    # Erstelle ZIP-Datei
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # JSON-Datei hinzufügen
        json_data = json.dumps(session.result, indent=2, ensure_ascii=False)
        zip_file.writestr("extraction_result.json", json_data)

        # Bilder hinzufügen (Pfade aus dem Ergebnis)
        for page in session.result.get("pages", []):
            for element in page.get("elements", []):
                if element.get("type") == "image" and element.get("file_path"):
                    try:
                        with open(element["file_path"], "rb") as img_file:
                            zip_file.writestr(
                                f"images/{Path(element['file_path']).name}",
                                img_file.read(),
                            )
                    except FileNotFoundError:
                        logger.warning(f"Bild nicht gefunden: {element['file_path']}")
                        continue

    zip_buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(zip_buffer.getvalue()),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=extraction_{session_id}.zip"
        },
    )


@app.post("/extract/email")
async def extract_pdf_with_email(
    file: UploadFile = File(...),
    email: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    """Startet Extraktion mit E-Mail-Benachrichtigung"""
    logger.info(f"Verarbeite E-Mail-Extraktion mit E-Mail: {email}")

    # Validierung der E-Mail
    if not email or "@" not in email:
        logger.error(f"E-Mail-Validierung fehlgeschlagen: {email}")
        raise HTTPException(status_code=400, detail="Ungültige E-Mail-Adresse")

    logger.info("E-Mail-Validierung erfolgreich, starte PDF-Extraktion")

    # Starte normale Extraktion
    response = await extract_pdf(file, background_tasks)
    logger.info(f"PDF-Extraktion gestartet mit Session ID: {response['session_id']}")

    # Füge E-Mail-Versand als separaten Background Task hinzu
    background_tasks.add_task(send_email_notification, response["session_id"], email)
    logger.info(
        f"E-Mail-Benachrichtigung Task hinzugefügt für Session: {response['session_id']}"
    )

    return {**response, "email": email, "message": "Ergebnis wird per E-Mail gesendet"}


async def process_pdf_extraction(file_path: str, session_id: str):
    """Background Task für PDF-Extraktion"""
    try:
        logger.info(f"process_pdf_extraction aufgerufen für Session: {session_id}")
        if session_id not in sessions:
            logger.error(f"Session {session_id} nicht gefunden in sessions")
            raise KeyError(f"Session {session_id} nicht gefunden")

        session = sessions[session_id]
        logger.info(f"Session gefunden, Status: {session.status}")
        session.status = "processing"
        logger.info(f"Status auf 'processing' gesetzt für Session: {session_id}")

        extractor = PDFExtractor(str(settings.output_dir))

        # Simuliere Fortschritt
        for progress in [10, 30, 50, 70, 90]:
            session.progress = progress
            await asyncio.sleep(0.5)

        # Führe Extraktion durch
        result = extractor.extract_pdf_data(file_path)
        session.result = result
        session.status = "completed"
        session.progress = 100

        logger.info(f"Extraktion abgeschlossen für Session {session_id}")

        # Bereinige temporäre Dateien
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Konnte temporäre Datei nicht bereinigen: {e}")

    except Exception as e:
        logger.error(
            f"Exception in process_pdf_extraction für Session {session_id}: {e}"
        )
        logger.error(f"Exception Type: {type(e).__name__}")

        # Überprüfe, ob die Session noch existiert
        if session_id in sessions:
            session = sessions[session_id]
            session.status = "error"
            session.error = str(e)
            logger.info(
                f"Session {session_id} Status auf 'error' gesetzt: {session.status}"
            )
        else:
            logger.error(
                f"Session {session_id} nicht mehr vorhanden für Fehlerbehandlung"
            )

        # Bereinige temporäre Dateien im Fehlerfall
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


async def send_email_notification(session_id: str, email: str):
    """Sendet E-Mail mit dem Extraktionsergebnis"""
    try:
        # Warte bis Extraktion abgeschlossen
        timeout = 300  # 5 Minuten Timeout
        start_time = asyncio.get_event_loop().time()

        while session_id in sessions and sessions[session_id].status not in [
            "completed",
            "error",
        ]:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.error(
                    f"Timeout bei E-Mail-Benachrichtigung für Session {session_id}"
                )
                return

            await asyncio.sleep(2)

        if session_id not in sessions:
            logger.error(f"Session {session_id} nicht gefunden für E-Mail-Versand")
            return

        session = sessions[session_id]
        if not session.result or session.status != "completed":
            logger.error(f"Extraktion für Session {session_id} fehlgeschlagen")
            return

        # Erstelle ZIP-Datei
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            json_data = json.dumps(session.result, indent=2, ensure_ascii=False)
            zip_file.writestr("extraction_result.json", json_data)

            for page in session.result.get("pages", []):
                for element in page.get("elements", []):
                    if element.get("type") == "image" and element.get("file_path"):
                        try:
                            with open(element["file_path"], "rb") as img_file:
                                zip_file.writestr(
                                    f"images/{Path(element['file_path']).name}",
                                    img_file.read(),
                                )
                        except FileNotFoundError:
                            continue

        # Sende E-Mail (hier nur Platzhalter)
        logger.info(f"Sende E-Mail an {email} mit Ergebnis für Session {session_id}")

        # In Produktion: E-Mail tatsächlich senden
        # await send_email_via_smtp(email, zip_buffer.getvalue())

        # Bereinige Ausgabedateien
        try:
            for page in session.result.get("pages", []):
                for element in page.get("elements", []):
                    if element.get("file_path"):
                        Path(element["file_path"]).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Konnte Ausgabedateien nicht bereinigen: {e}")

    except Exception as e:
        logger.error(f"Fehler beim E-Mail-Versand: {e}")


@app.get("/health")
async def health_check():
    """Health Check Endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions),
    }


@app.get("/sessions")
async def list_sessions():
    """Listet alle aktiven Sessions auf"""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "status": session.status,
                "progress": session.progress,
            }
            for session_id, session in sessions.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
