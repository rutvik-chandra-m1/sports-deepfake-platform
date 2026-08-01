# API Documentation

Base URL: `http://localhost:8000/api/v1`

Interactive docs (Swagger UI) are always available live at `/docs` once the backend is running.

## Health

### `GET /health`

Returns backend liveness/metadata.

```json
{
  "status": "ok",
  "app_name": "Sports Deepfake Detection & Verification Platform",
  "app_version": "0.1.0",
  "environment": "development"
}
```

## Media

### `POST /media/upload`

Upload an image or video (`multipart/form-data`, field name `file`). Validates extension
(against `ALLOWED_IMAGE_EXTENSIONS` / `ALLOWED_VIDEO_EXTENSIONS`) and size
(`MAX_UPLOAD_SIZE_MB`), streams it to `uploads/images/` or `uploads/videos/` under a unique
filename, creates a `pending` Analysis record pointing at it, and (Milestone 9) schedules the
full detection pipeline as a background task — poll `GET /analysis/{id}` to watch it move to
`processing` then `completed`/`failed` with a verdict.

Response: `201 Created` with the created `AnalysisRead` record — still `pending` at response
time (the pipeline runs in the background; the response doesn't wait for it).

Errors:
- `400` — unsupported file extension, or no file provided
- `413` — file exceeds the configured size limit

## Analysis

Backed by SQLite (`database/app.db`). As of Milestone 9, uploads automatically flow through the
full detection pipeline (Milestone 7 DL detector + Milestone 8 forensic signals, fused into a
verdict) via a background task — you don't need to call anything else after upload.

### `POST /analysis`

Create an analysis record directly (mainly useful for testing/demos without a real upload).

Request body (`AnalysisCreate`):
```json
{
  "filename": "interview_clip.mp4",
  "media_type": "video",          // "image" | "video"
  "status": "pending",             // "pending" | "processing" | "completed" | "failed" (default: pending)
  "verdict": null,                 // "authentic" | "suspicious" | null
  "confidence_score": null,        // 0-100 | null
  "risk_level": null,              // "low" | "medium" | "high" | null
  "explanation": null,
  "processing_duration_ms": null
}
```

Response: `201 Created` with the full `AnalysisRead` record (adds `id`, `detector_breakdown`,
`created_at`, `completed_at`). `explanation` (Milestone 10) is a natural-language verdict +
reasons report, not a raw score dump — see `docs/models.md`.

### `GET /analysis`

List analyses, newest first.

Query params: `offset` (default 0), `limit` (default 50), `search` (case-insensitive filename
substring match, Milestone 14), `verdict` (`authentic`/`suspicious`), `status`
(`pending`/`processing`/`completed`/`failed`). All filters are optional and combine with AND.

Response (`AnalysisList`):
```json
{ "total": 12, "items": [ /* AnalysisRead[] */ ] }
```

### `GET /analysis/{analysis_id}`

Fetch a single record. `404` if it doesn't exist.

### `POST /analysis/{analysis_id}/run`

(Milestone 9) Re-runs the full analysis pipeline on an existing record — useful for
reprocessing, or for records created via `POST /analysis` without a real upload. Sets status
back to `pending` and schedules the pipeline as a background task.

Response: `202 Accepted` with the record (still `pending` at response time — poll `GET
/analysis/{id}` to watch it complete). `400` if the record has no `file_path`. `404` if it
doesn't exist.

### `DELETE /analysis/{analysis_id}`

Delete a record. `204 No Content` on success, `404` if it doesn't exist.

---

## Planned (not yet implemented)

| Endpoint | Milestone |
|---|---|
| `GET /analysis/{analysis_id}/status` — lightweight polling endpoint (status only, no full record) | later, if needed |
