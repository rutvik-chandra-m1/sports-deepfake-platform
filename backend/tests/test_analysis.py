from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "filename": "interview_clip.mp4",
    "media_type": "video",
    "status": "completed",
    "verdict": "suspicious",
    "confidence_score": 87.5,
    "risk_level": "high",
    "explanation": "Temporal inconsistency detected around frame 214.",
    "processing_duration_ms": 4300,
}


def test_create_analysis_returns_created_record(client: TestClient):
    response = client.post("/api/v1/analysis", json=VALID_PAYLOAD)
    assert response.status_code == 201

    body = response.json()
    assert body["id"] is not None
    assert body["filename"] == VALID_PAYLOAD["filename"]
    assert body["verdict"] == "suspicious"
    assert body["confidence_score"] == 87.5
    assert body["created_at"] is not None


def test_create_analysis_defaults_status_to_pending(client: TestClient):
    minimal_payload = {"filename": "raw_upload.jpg", "media_type": "image"}
    response = client.post("/api/v1/analysis", json=minimal_payload)
    assert response.status_code == 201
    assert response.json()["status"] == "pending"


def test_list_analyses_returns_created_records_newest_first(client: TestClient):
    first = client.post("/api/v1/analysis", json=VALID_PAYLOAD).json()
    second_payload = {**VALID_PAYLOAD, "filename": "second_clip.mp4"}
    second = client.post("/api/v1/analysis", json=second_payload).json()

    response = client.get("/api/v1/analysis")
    assert response.status_code == 200

    body = response.json()
    assert body["total"] >= 2
    ids_in_order = [item["id"] for item in body["items"]]
    assert ids_in_order.index(second["id"]) < ids_in_order.index(first["id"])


def test_get_analysis_by_id(client: TestClient):
    created = client.post("/api/v1/analysis", json=VALID_PAYLOAD).json()

    response = client.get(f"/api/v1/analysis/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_missing_analysis_returns_404(client: TestClient):
    response = client.get("/api/v1/analysis/999999")
    assert response.status_code == 404


def test_delete_analysis_removes_record(client: TestClient):
    created = client.post("/api/v1/analysis", json=VALID_PAYLOAD).json()

    delete_response = client.delete(f"/api/v1/analysis/{created['id']}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/api/v1/analysis/{created['id']}")
    assert get_response.status_code == 404


def test_delete_missing_analysis_returns_404(client: TestClient):
    response = client.delete("/api/v1/analysis/999999")
    assert response.status_code == 404


def test_list_analyses_filters_by_search(client: TestClient):
    unique = "zzqquniquemarker"
    match = client.post(
        "/api/v1/analysis", json={**VALID_PAYLOAD, "filename": f"{unique}_clip.mp4"}
    ).json()
    client.post("/api/v1/analysis", json={**VALID_PAYLOAD, "filename": "unrelated_clip.mp4"})

    response = client.get(f"/api/v1/analysis?search={unique}")
    body = response.json()

    ids = [item["id"] for item in body["items"]]
    assert ids == [match["id"]]


def test_list_analyses_filters_by_verdict(client: TestClient):
    unique = "verdictfiltermarker"
    authentic = client.post(
        "/api/v1/analysis",
        json={**VALID_PAYLOAD, "filename": f"{unique}_authentic.mp4", "verdict": "authentic"},
    ).json()
    suspicious = client.post(
        "/api/v1/analysis",
        json={**VALID_PAYLOAD, "filename": f"{unique}_suspicious.mp4", "verdict": "suspicious"},
    ).json()

    response = client.get(f"/api/v1/analysis?search={unique}&verdict=authentic")
    ids = [item["id"] for item in response.json()["items"]]

    assert authentic["id"] in ids
    assert suspicious["id"] not in ids


def test_list_analyses_filters_by_status(client: TestClient):
    unique = "statusfiltermarker"
    completed = client.post(
        "/api/v1/analysis",
        json={**VALID_PAYLOAD, "filename": f"{unique}_completed.mp4", "status": "completed"},
    ).json()
    pending = client.post(
        "/api/v1/analysis",
        json={"filename": f"{unique}_pending.mp4", "media_type": "video"},
    ).json()

    response = client.get(f"/api/v1/analysis?search={unique}&status=pending")
    ids = [item["id"] for item in response.json()["items"]]

    assert pending["id"] in ids
    assert completed["id"] not in ids


def test_list_analyses_search_is_case_insensitive(client: TestClient):
    unique = "CaseMarkerXYZ"
    created = client.post(
        "/api/v1/analysis", json={**VALID_PAYLOAD, "filename": f"{unique}.mp4"}
    ).json()

    response = client.get(f"/api/v1/analysis?search={unique.lower()}")
    ids = [item["id"] for item in response.json()["items"]]

    assert created["id"] in ids
