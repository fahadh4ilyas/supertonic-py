"""HTTP route handlers for ``supertonic serve``.

The route surface is intentionally narrow and follows two conventions so that
existing clients work with minimal changes:

1. **Native namespace** under ``/v1/...`` for first-class Supertonic features.
2. **OpenAI Audio Speech alias** at ``POST /v1/audio/speech`` so any client
   that already speaks the OpenAI API (n8n OpenAI node, openai-python, many
   browser extensions, Electron tools) can swap the base URL.

Errors use the OpenAI-shaped envelope::

    { "error": { "message": "...", "type": "...", "code": "..." } }

so that downstream error parsers keep working.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF
from fastapi import APIRouter, FastAPI, File, Form, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import AVAILABLE_LANGUAGES, DEFAULT_SILENCE_DURATION, get_custom_voices_dir
from ..core import Style
from . import styles_store, voice_manager
from .audio import (
    SUPPORTED_FORMATS,
    UnsupportedAudioFormat,
    coerce_response_format,
    duration_seconds,
    encode_audio,
    format_to_mime,
)
from .schemas import (
    BatchRequest,
    BatchResponse,
    BatchResultItem,
    ErrorDetail,
    ErrorEnvelope,
    HealthResponse,
    OpenAISpeechRequest,
    StyleImportResponse,
    StyleInfo,
    StylesResponse,
    TTSRequest,
)

if TYPE_CHECKING:
    from .app import ServerState

logger = logging.getLogger(__name__)


# Maximum body size accepted by ``POST /v1/styles/import``. The bundled
# voice-style JSONs are ~290 kB; 1 MiB leaves comfortable headroom while
# bounding any single request's memory footprint. Enforced twice:
#
# 1. as a middleware-level ``Content-Length`` pre-flight check (so an
#    oversized request is rejected before the body is buffered), and
# 2. inside the import handler via ``file.read(MAX+1)`` (a fallback for
#    chunked transfer encoding where ``Content-Length`` is absent).
MAX_STYLE_IMPORT_BYTES = 1 * 1024 * 1024


class UnknownVoice(LookupError):
    """Voice name does not match any built-in or imported style."""


def _state(request: Request) -> "ServerState":
    return request.app.state.server_state  # type: ignore[no-any-return]


def _error(status_code: int, message: str, code: str, type_: str = "invalid_request_error"):
    env = ErrorEnvelope(error=ErrorDetail(message=message, type=type_, code=code))
    return JSONResponse(status_code=status_code, content=env.model_dump())


def _resolve_voice(state: "ServerState", voice_name: str):
    """Return a ``Style`` for ``voice_name`` from built-ins, uploaded, or imported styles."""
    tts = state.tts
    if tts is None:
        raise RuntimeError("server not ready")
    if voice_name in tts.voice_style_names:
        return tts.get_voice_style(voice_name)
    # Check uploaded voices (encoder-extracted styles)
    if hasattr(tts, "has_encoder") and tts.has_encoder():
        voices_dir = get_custom_voices_dir()
        result = voice_manager.load_voice(voices_dir, voice_name)
        if result is not None:
            style_ttl_np = result[0].cpu().numpy()
            style_dp_np = result[1].cpu().numpy()
            return Style(style_ttl_onnx=style_ttl_np, style_dp_onnx=style_dp_np)
    custom_path = state.custom_styles.get(voice_name)
    if custom_path is not None:
        return tts.get_voice_style_from_path(custom_path)
    raise UnknownVoice(voice_name)


def _do_synthesize(
    state: "ServerState",
    *,
    text: str,
    voice: str,
    lang: Optional[str],
    speed: Optional[float],
    steps: Optional[int],
    max_chunk_length: Optional[int],
    silence_duration: Optional[float],
) -> tuple[np.ndarray, float]:
    """Resolve voice, take the synth lock, run synthesize, return (wav, duration_s)."""
    style = _resolve_voice(state, voice)
    kwargs: dict[str, Any] = {"voice_style": style}
    if speed is not None:
        kwargs["speed"] = speed
    if steps is not None:
        kwargs["total_steps"] = steps
    if max_chunk_length is not None:
        kwargs["max_chunk_length"] = max_chunk_length
    if silence_duration is not None:
        kwargs["silence_duration"] = silence_duration
    if lang is not None:
        kwargs["lang"] = lang

    # ONNX Runtime sessions are not thread-safe — serialize access.
    # PyTorch handles concurrency natively, so the lock is skipped.
    if state.use_onnx:
        with state.synth_lock:
            wav, _ = state.tts.synthesize(text=text, **kwargs)
    else:
        wav, _ = state.tts.synthesize(text=text, **kwargs)

    return wav, duration_seconds(wav, state.tts.sample_rate)


def _audio_response(state: "ServerState", wav: np.ndarray, fmt: str, duration_s: float) -> Response:
    body = encode_audio(wav, state.tts.sample_rate, fmt)
    return Response(
        content=body,
        media_type=format_to_mime(fmt),
        headers={
            "X-Audio-Duration": f"{duration_s:.3f}",
            "X-Supertonic-Version": __version__,
            "X-Sample-Rate": str(state.tts.sample_rate),
        },
    )


def _validate_lang(lang: Optional[str]):
    if lang is not None and lang not in AVAILABLE_LANGUAGES:
        return _error(
            400,
            f"unsupported lang {lang!r}; valid: {', '.join(AVAILABLE_LANGUAGES)}",
            "unsupported_lang",
        )
    return None


def register_routes(app: FastAPI) -> None:
    """Attach all `/v1/...` routes to ``app``.

    Called from :func:`supertonic.server.app.create_app` after the lifespan and
    ``app.state.server_state`` have been set up.
    """
    router = APIRouter()

    @router.get("/v1/health", response_model=HealthResponse)
    def health(request: Request):
        state = _state(request)
        if not state.is_ready or state.tts is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="loading",
                    model=state.model,
                    version=__version__,
                    voices_loaded=0,
                ).model_dump(),
            )
        return HealthResponse(
            status="ok",
            model=state.model,
            sample_rate=state.tts.sample_rate,
            version=__version__,
            voices_loaded=len(state.tts.voice_style_names) + len(state.custom_styles) + len(voice_manager.list_voices(get_custom_voices_dir())),
        )

    @router.get("/v1/styles", response_model=StylesResponse)
    def list_styles(request: Request):
        state = _state(request)
        if state.tts is None:
            return _error(503, "server not ready", "not_ready", type_="server_error")
        builtin = [StyleInfo(name=n, kind="builtin") for n in state.tts.voice_style_names]
        custom = [
            StyleInfo(name=n, kind="custom", path=str(p))
            for n, p in sorted(state.custom_styles.items())
        ]
        return StylesResponse(styles=builtin + custom)

    @router.post("/v1/styles/import", response_model=StyleImportResponse)
    async def import_style(
        request: Request,
        overwrite: bool = False,
        file: Optional[UploadFile] = File(None),
        name: Optional[str] = Form(None),
    ):
        state = _state(request)
        if state.tts is None:
            return _error(503, "server not ready", "not_ready", type_="server_error")

        ct = request.headers.get("content-type", "")
        chosen_name: Optional[str]
        if ct.startswith("multipart/form-data"):
            if file is None:
                return _error(400, "missing 'file' part", "missing_file")
            # Read with an explicit cap as a fallback for chunked uploads
            # that bypass the middleware's Content-Length pre-flight check.
            raw = await file.read(MAX_STYLE_IMPORT_BYTES + 1)
            if len(raw) > MAX_STYLE_IMPORT_BYTES:
                return _error(
                    413,
                    f"uploaded voice style exceeds {MAX_STYLE_IMPORT_BYTES} bytes",
                    "payload_too_large",
                )
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                return _error(400, f"invalid JSON in uploaded file: {e}", "invalid_json")
            chosen_name = name or Path(file.filename or "").stem or "imported"
        else:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return _error(400, "invalid JSON body", "invalid_json")
            if not isinstance(body, dict):
                return _error(400, "JSON body must be an object", "invalid_body")
            chosen_name = body.get("name")
            if not chosen_name:
                return _error(400, "missing 'name' in JSON body", "missing_name")
            data = {k: body[k] for k in ("style_ttl", "style_dp") if k in body}

        try:
            target = styles_store.save(
                state.custom_styles_dir,
                chosen_name,
                data,
                builtin_names=state.tts.voice_style_names,
                overwrite=overwrite,
            )
        except styles_store.InvalidStyleName as e:
            return _error(400, str(e), "invalid_style_name")
        except styles_store.StyleNameConflict as e:
            status = 409 if "already exists" in str(e) else 400
            return _error(status, str(e), "style_name_conflict")
        except ValueError as e:
            return _error(400, str(e), "invalid_style_payload")

        state.custom_styles[target.stem] = target
        return StyleImportResponse(name=target.stem, stored_at=str(target))

    @router.post("/v1/tts")
    def synth_native(req: TTSRequest, request: Request):
        state = _state(request)
        if state.tts is None:
            return _error(503, "server not ready", "not_ready", type_="server_error")
        try:
            fmt = coerce_response_format(req.response_format)
        except UnsupportedAudioFormat as e:
            return _error(
                400,
                f"unsupported response_format {str(e)!r}",
                "unsupported_response_format",
            )
        err = _validate_lang(req.lang)
        if err is not None:
            return err
        try:
            wav, dur = _do_synthesize(
                state,
                text=req.text,
                voice=req.voice,
                lang=req.lang,
                speed=req.speed,
                steps=req.steps,
                max_chunk_length=req.max_chunk_length,
                silence_duration=req.silence_duration,
            )
        except UnknownVoice as e:
            return _error(400, f"unknown voice {str(e)!r}", "unknown_voice")
        except Exception as e:  # noqa: BLE001 — surface as 500 with code
            logger.exception("synthesis failed")
            return _error(500, f"synthesis failed: {e}", "synthesis_failed", type_="server_error")
        return _audio_response(state, wav, fmt, dur)

    @router.post("/v1/audio/speech")
    def openai_compat_speech(req: OpenAISpeechRequest, request: Request):
        # Validate ``model`` against AVAILABLE_MODELS but only *accept* the
        # model currently loaded — switching at request time is out of scope.
        state = _state(request)
        if req.model not in OpenAISpeechRequest.valid_models():
            return _error(
                400,
                f"unknown model {req.model!r}; valid: {', '.join(OpenAISpeechRequest.valid_models())}",
                "unknown_model",
            )
        if req.model != state.model:
            return _error(
                400,
                f"this server serves {state.model!r}; request asked for {req.model!r}. "
                f"Restart with --model {req.model} to switch.",
                "model_not_loaded",
            )
        if state.tts is None:
            return _error(503, "server not ready", "not_ready", type_="server_error")
        # OpenAI clients default to ``response_format='mp3'`` — surface a
        # clear error rather than silently emitting WAV.
        try:
            fmt = coerce_response_format(req.response_format)
        except UnsupportedAudioFormat as e:
            return _error(
                400,
                f"unsupported response_format {str(e)!r}; "
                f"set response_format to one of: {', '.join(SUPPORTED_FORMATS)}",
                "unsupported_response_format",
            )
        err = _validate_lang(req.lang)
        if err is not None:
            return err
        try:
            wav, dur = _do_synthesize(
                state,
                text=req.input,
                voice=req.voice,
                lang=req.lang,
                speed=req.speed,
                steps=None,
                max_chunk_length=None,
                silence_duration=None,
            )
        except UnknownVoice as e:
            return _error(400, f"unknown voice {str(e)!r}", "unknown_voice")
        except Exception as e:  # noqa: BLE001
            logger.exception("synthesis failed")
            return _error(500, f"synthesis failed: {e}", "synthesis_failed", type_="server_error")
        return _audio_response(state, wav, fmt, dur)

    @router.post("/v1/tts/batch", response_model=BatchResponse)
    def synth_batch(req: BatchRequest, request: Request):
        state = _state(request)
        if state.tts is None:
            return _error(503, "server not ready", "not_ready", type_="server_error")
        try:
            fmt = coerce_response_format(req.response_format)
        except UnsupportedAudioFormat as e:
            return _error(
                400,
                f"unsupported response_format {str(e)!r}",
                "unsupported_response_format",
            )
        defaults = req.defaults
        results: list[BatchResultItem] = []
        for idx, item in enumerate(req.items):
            voice = item.voice or (defaults.voice if defaults else None) or "M1"
            lang = item.lang or (defaults.lang if defaults else None)
            speed = item.speed if item.speed is not None else (defaults.speed if defaults else None)
            steps = item.steps if item.steps is not None else (defaults.steps if defaults else None)
            mcl = (
                item.max_chunk_length
                if item.max_chunk_length is not None
                else (defaults.max_chunk_length if defaults else None)
            )
            sil = (
                item.silence_duration
                if item.silence_duration is not None
                else (defaults.silence_duration if defaults else None)
            )
            if lang is not None and lang not in AVAILABLE_LANGUAGES:
                return _error(
                    400,
                    f"items[{idx}].lang: unsupported lang {lang!r}",
                    "unsupported_lang",
                )
            try:
                wav, dur = _do_synthesize(
                    state,
                    text=item.text,
                    voice=voice,
                    lang=lang,
                    speed=speed,
                    steps=steps,
                    max_chunk_length=mcl,
                    silence_duration=sil,
                )
            except UnknownVoice as e:
                return _error(
                    400,
                    f"items[{idx}]: unknown voice {str(e)!r}",
                    "unknown_voice",
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("batch item %d synthesis failed", idx)
                return _error(
                    500,
                    f"items[{idx}]: synthesis failed: {e}",
                    "synthesis_failed",
                    type_="server_error",
                )
            body = encode_audio(wav, state.tts.sample_rate, fmt)
            results.append(
                BatchResultItem(
                    audio_base64=base64.b64encode(body).decode("ascii"),
                    duration_s=dur,
                    format=fmt,
                    sample_rate=state.tts.sample_rate,
                )
            )
        return BatchResponse(items=results)

    @router.get("/v1/audio/voices")
    def openai_compat_voices(request: Request):
        """OpenAI/vLLM-Omni compatible endpoint to list available voices."""
        state = _state(request)
        if state.tts is None:
            return _error(503, "server not ready", "not_ready", type_="server_error")

        builtin_voices = list(state.tts.voice_style_names)
        custom_voices = list(state.custom_styles.keys())

        # Uploaded voices (encoder-extracted, from voice_manager)
        uploaded_meta = voice_manager.list_voices(get_custom_voices_dir())
        uploaded_names = [v["name"] for v in uploaded_meta]

        all_voices = builtin_voices + custom_voices + uploaded_names

        uploaded_voices = []
        for name in custom_voices:
            uploaded_voices.append({
                "name": name,
                "speaker_description": "Supertonic custom imported style",
            })
        uploaded_voices.extend(uploaded_meta)

        return JSONResponse(content={
            "voices": sorted(all_voices),
            "uploaded_voices": uploaded_voices,
        })

    # Map friendly OpenAI language names to Supertonic ISO codes
    LANG_MAP = {
        "arabic": "ar",
        "bulgarian": "bg",
        "croatian": "hr",
        "czech": "cs",
        "danish": "da",
        "dutch": "nl",
        "english": "en",
        "estonian": "et",
        "finnish": "fi",
        "french": "fr",
        "german": "de",
        "greek": "el",
        "hindi": "hi",
        "hungarian": "hu",
        "indonesian": "id",
        "italian": "it",
        "japanese": "ja",
        "korean": "ko",
        "latvian": "lv",
        "lithuanian": "lt",
        "polish": "pl",
        "portuguese": "pt",
        "romanian": "ro",
        "russian": "ru",
        "slovak": "sk",
        "slovenian": "sl",
        "spanish": "es",
        "swedish": "sv",
        "turkish": "tr",
        "ukrainian": "uk",
        "vietnamese": "vi",
        "auto": "na"
    }

    def _map_language(lang_str: str) -> str:
        if not lang_str:
            return "na"
        lang_str_lower = lang_str.lower()
        if lang_str_lower in LANG_MAP:
            return LANG_MAP[lang_str_lower]
        return lang_str
    
    @router.websocket("/v1/audio/speech/stream")
    async def openai_compat_speech_stream(websocket: WebSocket):
        await websocket.accept()
        
        state = websocket.app.state.server_state
        if state.tts is None:
            await websocket.send_json({"type": "error", "message": "server not ready"})
            await websocket.close(code=1011)
            return

        config = {}
        text_buffer = ""
        sentence_queue = asyncio.Queue()

        # --- BACKGROUND WORKER TASK ---
        async def tts_worker():
            sentence_index = 0
            total_sentences = 0
            try:
                while True:
                    # Wait for the next sentence from the queue
                    sentence_text = await sentence_queue.get()
                    
                    # A 'None' value is our sentinel signal to stop the worker
                    if sentence_text is None:
                        break

                    try:
                        mapped_lang = _map_language(config.get("language", "auto").lower())

                        if mapped_lang not in AVAILABLE_LANGUAGES:
                            await websocket.send_json({"type": "error", "message": f"unsupported lang {mapped_lang!r}"})
                            continue
                            
                        # 1. We send the start signal immediately so the frontend knows audio is coming
                        await websocket.send_json({
                            "type": "audio.start",
                            "sentence_index": sentence_index,
                            "sentence_text": sentence_text,
                            "format": "pcm",
                            "sample_rate": state.tts.sample_rate
                        })

                        # 2. Resolve voice (built-in, uploaded, or custom) and create generator
                        style = _resolve_voice(state, config.get("voice", "M1"))

                        if state.use_onnx:
                            with state.synth_lock:
                                chunk_generator = state.tts.synthesize_generator(
                                    text=sentence_text,
                                    voice_style=style,
                                    lang=mapped_lang,
                                    speed=config.get("speed", 1.0),
                                    max_chunk_length=config.get("max_chunk_length"),
                                    silence_duration=config.get("silence_duration", DEFAULT_SILENCE_DURATION),
                                )
                        else:
                            chunk_generator = state.tts.synthesize_generator(
                                text=sentence_text,
                                voice_style=style,
                                lang=mapped_lang,
                                speed=config.get("speed", 1.0),
                                max_chunk_length=config.get("max_chunk_length"),
                                silence_duration=config.get("silence_duration", DEFAULT_SILENCE_DURATION),
                            )

                        total_len_pcm_data = 0

                        # 3. Iterate chunks (locked for ONNX, concurrent for PyTorch)
                        while True:
                            if state.use_onnx:
                                def _locked_next():
                                    with state.synth_lock:
                                        return next(chunk_generator, None)
                                chunk_data = await asyncio.to_thread(_locked_next)
                            else:
                                chunk_data = await asyncio.to_thread(next, chunk_generator, None)
                            
                            if chunk_data is None:
                                # The generator returned None, meaning this sentence is fully spoken
                                break

                            wav, dur = chunk_data
                                
                            # Convert the chunk and instantly send it to the WebRTC socket
                            pcm_data = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                            await websocket.send_bytes(pcm_data)
                            total_len_pcm_data += len(pcm_data)

                        # 4. Notify the frontend the sentence is done
                        await websocket.send_json({
                            "type": "audio.done",
                            "sentence_index": sentence_index,
                            "total_bytes": total_len_pcm_data,
                            "error": False
                        })

                        sentence_index += 1
                        total_sentences += 1

                    except Exception as e:
                        logger.exception("Streaming synthesis failed")
                        await websocket.send_json({"type": "error", "message": str(e)})

                    finally:
                        # Tell the queue we finished processing this item
                        sentence_queue.task_done()
                        
                # Only send session.done after the queue is completely empty and finished
                await websocket.send_json({
                    "type": "session.done",
                    "total_sentences": total_sentences
                })

            except asyncio.CancelledError:
                # Worker was killed (e.g. client disconnected early)
                pass


        # Start the background worker immediately
        worker_task = asyncio.create_task(tts_worker())

        # --- MAIN WEBSOCKET LISTENER ---
        try:
            while True:
                msg = await websocket.receive_json()
                msg_type = msg.get("type")

                if msg_type == "session.config":
                    config = msg
                    
                elif msg_type == "input.text":
                    text_buffer += msg.get("text", "")
                    
                    # Split sentences dynamically
                    split_commas = config.get("split_commas", False)
                    if split_commas:
                        sentences = re.split(r'(?<=[.!?。！？,，\n])(?=\s|$)', text_buffer)
                    else:
                        sentences = re.split(r'(?<=[.!?。！？\n])(?=\s|$)', text_buffer)
                    
                    if len(sentences) > 1:
                        for s in sentences[:-1]:
                            if s.strip():
                                # Instantly put the sentence in the queue instead of blocking
                                await sentence_queue.put(s.strip())
                        text_buffer = sentences[-1]
                        
                elif msg_type == "input.done":
                    if text_buffer.strip():
                        await sentence_queue.put(text_buffer.strip())
                        text_buffer = ""
                    
                    # Send the sentinel 'None' to tell the worker we are done sending text
                    await sentence_queue.put(None)
                    
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected gracefully.")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up: If the client disconnects or an error occurs, kill the TTS worker
            worker_task.cancel()

    app.include_router(router)


def include_voice_routes(app: FastAPI) -> None:
    """Conditionally include voice management routes (PyTorch backend only)."""
    voice_router = APIRouter()

    @voice_router.post("/v1/audio/voices")
    async def upload_voice(
        request: Request,
        audio_sample: UploadFile = File(...),
        consent: str = Form(...),
        name: str = Form(...),
        ref_text: Optional[str] = Form(None),
        speaker_description: Optional[str] = Form(None),
    ):
        """Upload a voice sample for voice cloning.

        Requires PyTorch backend with a trained AudioEncoder.
        """
        state = _state(request)
        if state.tts is None:
            return JSONResponse(status_code=503, content={"error": "server not ready"})

        if state.use_onnx or not getattr(state.tts, "has_encoder", lambda: False)():
            return JSONResponse(
                status_code=400,
                content={"error": "Voice upload requires PyTorch backend with a trained AudioEncoder. "
                                  "Start with --no-use-onnx and call model.load_encoder()."},
            )

        audio_bytes = await audio_sample.read()
        file_size = len(audio_bytes)

        try:
            wav, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Unsupported audio format"})

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        target_sr = state.tts.sample_rate
        if sr != target_sr:
            wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
            wav_t = taF.resample(wav_t, sr, target_sr)
            wav = wav_t.squeeze(0).numpy()

        device = next(state.tts.parameters()).device
        waveform = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            style_ttl, style_dp = state.tts.audio_encoder(waveform)

        voices_dir = get_custom_voices_dir()
        meta = voice_manager.save_voice(
            voices_dir, name,
            style_ttl=style_ttl,
            style_dp=style_dp,
            consent=consent,
            ref_text=ref_text,
            speaker_description=speaker_description,
            mime_type=audio_sample.content_type or "audio/wav",
            file_size=file_size,
        )

        return {"success": True, "voice": meta}

    @voice_router.delete("/v1/audio/voices/{name}")
    async def delete_voice(name: str, request: Request):
        """Delete an uploaded voice."""
        state = _state(request)
        if state.tts is None:
            return JSONResponse(status_code=503, content={"error": "server not ready"})

        voices_dir = get_custom_voices_dir()
        deleted = voice_manager.delete_voice(voices_dir, name)
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Voice '{name}' not found"},
            )
        return {"success": True, "message": f"Voice '{name}' deleted successfully"}

    app.include_router(voice_router)
