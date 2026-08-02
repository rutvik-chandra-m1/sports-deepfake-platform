# Security

Status of every finding from the engineering review, what was done, and what is still open.
Regression tests live in `backend/tests/test_security.py` — each asserts on an *attacker outcome*
("the file was not read", "the path is not disclosed"), so a refactor that preserves the property
passes and one that loses it fails.

---

## Fixed in R9

### 1. Arbitrary server-side file read — **HIGH**

**Was:** `AnalysisCreate.file_path` was a client-settable field, and the analysis pipeline passed
it straight to `cv2.imread` / `cv2.VideoCapture`. A client could name **any** path on the server
and have its contents analysed, or probe the filesystem through the error text stored on the
record and returned by the API.

**Fixed, two independent layers:**
1. `file_path` was removed from the public `AnalysisCreate` schema entirely. A separate
   `AnalysisCreateInternal` carries it, and only `storage_service.save_upload()` — which chooses
   the location itself — constructs one.
2. `assert_within_upload_dir()` re-verifies containment at the point the file is actually opened,
   rather than trusting whatever wrote the record. It compares **resolved** paths, so `..`
   traversal, symlinks and mixed separators all collapse to a real location first; a
   `startswith()` check would miss a symlink pointing out of the directory.

Rejections are logged with the attempted path server-side but the client sees only *"the stored
file path is not permitted"* — echoing the path back would confirm filesystem layout to whoever
probed it.

**Verified live:** posting `file_path: "C:/Windows/win.ini"` is accepted as a record (the unknown
field is ignored) but `POST /analysis/{id}/run` returns **400** — nothing is read.

### 2. Internal path disclosure — **MEDIUM**

**Was:** every API response carried the absolute server path of the stored file. The UI never
displayed it.

**Fixed:** `file_path` removed from `AnalysisRead`. **Verified live:** absent from both upload and
detail responses.

### 3. Extension-only upload validation — **MEDIUM**

**Was:** uploads were accepted on file extension alone. Anything could be renamed `.jpg` and
handed to OpenCV's C++ decoders — historically a rich source of memory-safety bugs.

**Fixed:** `assert_content_matches_extension()` checks real magic bytes for every accepted format
(JPEG, PNG, WebP, MP4, MOV, AVI, MKV) against the **first chunk**, before the rest is committed to
disk. Rejected uploads are deleted, never left behind.

Deliberately hand-written rather than pulling in `python-magic`/`filetype`: the check is small
enough to audit directly, and every third-party package on a security path is another
supply-chain surface. It is a format guard, **not** a malware scanner.

**Verified live:** a ZIP renamed to `.jpg` → **400**; a genuine JPEG → **201**.

### 4. No rate limiting / unbounded pagination — **MEDIUM**

**Was:** no limits at all. `?limit=1000000` was a valid query, and uploads (which run the full
detection pipeline) could be issued without restriction.

**Fixed:** a fixed-memory sliding-window limiter keyed by client IP, with **two budgets** —
uploads are far more expensive than reads and get a tighter one (10/min vs 60/min, both
configurable). Page size is clamped to `max_page_size` (200) rather than rejected, so an
over-large request returns the maximum page instead of an error.

`X-Forwarded-For` is honoured **only** when `trust_proxy_headers` is explicitly enabled — the
header is client-settable, so trusting it by default would let anyone spoof identity and bypass
the limiter.

### 5. Unsafe defaults reaching production — **MEDIUM**

**Was:** `debug=True` and `secret_key="change-me-in-production"` shipped as defaults with nothing
checking them.

**Fixed:** `validate_production_settings()` runs at startup and **refuses to boot** when
`APP_ENV` is not a development/test value and any of these hold: the placeholder secret is still
set, `API_KEY` is empty, `DEBUG` is true, or `CORS_ORIGINS` contains `*` while credentials are
allowed. Failing loudly at boot beats serving traffic with the shipped placeholder secret.

### 6. No authentication — **HIGH (partially addressed)**

**Was:** every endpoint open, including `DELETE /analysis/{id}`.

**Now:** a shared API key (`X-API-Key`) guards the media and analysis routers, compared with
`hmac.compare_digest` so key material does not leak through timing. Enforced whenever `API_KEY`
is set; **startup refuses production without one**, so "unset" can never silently mean "open" in
deployment. Left optional in development so local work and the test suite are unaffected.

**This is deliberately a shared key, not per-user auth** — see *Still open* below.

### 7. Missing security headers — **LOW**

**Fixed:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `Permissions-Policy`, and a restrictive
`Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` (this API serves only JSON
to a separate SPA origin, so a strict CSP costs nothing). **Verified live.**

---

## Still open — known and deliberate

| # | Finding | Severity | Why not now |
|---|---|---|---|
| 8 | **No per-user accounts or ownership.** A single shared key authenticates *the client*, not *a user*; anyone holding it sees and can delete every analysis. | Medium | Needs a `User` model, sessions and per-record ownership — properly R10/R11 territory, and multi-tenancy was never in scope for a single-user academic demo. |
| 9 | **Model supply chain unverified.** Weights are pulled from a third-party Hugging Face account with no pinned revision SHA and no checksum; the Haar cascade and MediaPipe bundle are fetched over bare `urllib` with no integrity check. | Medium | Pin a revision and record hashes. Cheap, not yet done. |
| 10 | **Uploaded files are never deleted.** Deleting an analysis orphans its media; there is no retention policy on user-submitted content. | Low–Medium | Real privacy concern for a tool ingesting photos of identifiable people. Belongs with R10's cleanup job. |
| 11 | **Rate limiter is in-process.** Correct for one uvicorn process; useless behind a load balancer. | Low | Replace with a shared store *if* the service is ever scaled horizontally. |
| 12 | **`allow_credentials=True` with `allow_methods/headers=["*"]`.** Safe today because origins are explicit, and startup now rejects `*` origins in production. | Low | Tighten to the methods actually used. |
| 13 | **No HTTPS enforcement.** | Low | Belongs at the reverse proxy (R14 Docker/nginx), not in application code. |

---

## Configuration

```bash
# Development (defaults) -- auth disabled, warned about at startup
APP_ENV=development

# Production -- startup FAILS if any of these are missing/unsafe
APP_ENV=production
SECRET_KEY=<generate a real one>
API_KEY=<generate a real one>
DEBUG=false
CORS_ORIGINS=https://your-frontend.example
TRUST_PROXY_HEADERS=true      # only when genuinely behind a trusted proxy
```

Tuning: `RATE_LIMIT_REQUESTS` (60/min), `UPLOAD_RATE_LIMIT_REQUESTS` (10/min),
`MAX_PAGE_SIZE` (200).

---

## Threat model, briefly

**In scope:** an unauthenticated network client reaching the API — reading files it shouldn't,
learning the filesystem layout, feeding hostile bytes to native decoders, exhausting resources,
or reaching a production instance still running development defaults.

**Out of scope:** a hostile operator (they own the machine), and adversarial ML attacks against
the detector itself — an attacker who can craft images specifically to fool the model is a real
threat for a detection product, but it is a *model* robustness problem, not an application
security one, and is unaddressed. The PPT names adversarial attacks explicitly; see
`docs/evaluation.md` for what the detector's measured limits actually are.
