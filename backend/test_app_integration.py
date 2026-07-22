"""
Test Flask backend routes and integration with frontend static files.
"""
import sys
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import app

def run_tests():
    print("=== Testing Flask App Endpoints & Integration ===")
    client = app.test_client()

    # 1. Test Health Endpoint
    resp = client.get("/health")
    print(f"GET /health -> Status: {resp.status_code}, Data: {resp.get_json()}")
    assert resp.status_code == 200, "Health check failed!"

    # 2. Test Inputs List Endpoint
    resp = client.get("/inputs")
    print(f"GET /inputs -> Status: {resp.status_code}, Data: {resp.get_json()}")
    assert resp.status_code == 200, "Inputs route failed!"

    # 3. Test Static Index Endpoint
    resp = client.get("/")
    print(f"GET / -> Status: {resp.status_code}, Content-Type: {resp.content_type}, Size: {len(resp.data)} bytes")
    assert resp.status_code == 200, "Index HTML route failed!"
    assert b"HALO" in resp.data, "Index page HTML content invalid!"

    # 4. Test About Endpoint
    resp = client.get("/about")
    print(f"GET /about -> Status: {resp.status_code}")
    assert resp.status_code == 200, "About HTML route failed!"

    print("\n=== Flask Backend & Frontend Integration Tests Passed Successfully! ===")

if __name__ == "__main__":
    run_tests()
