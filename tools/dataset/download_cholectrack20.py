"""
Download CholecTrack20 Dataset.

Source:
https://www.synapse.org/Synapse:syn53182642/wiki/628404
"""

import os
import sys
from pathlib import Path

import requests
import synapseclient
import synapseutils


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    email = os.environ["SYNAPSE_EMAIL"]
    auth_token = os.environ["SYNAPSE_AUTH_TOKEN"]
    access_key = os.environ["CHOLECTRACK20_ACCESS_KEY"]
    local_folder = Path(
        os.environ.get("CHOLECTRACK20_DATASET_ROOT", "D:/cholec_dataset")
    )

    # 1. Login with Synapse credentials
    print("Authenticating user ...")
    syn = synapseclient.login(email=email, authToken=auth_token)

    # 2. Request entity access
    print("Authenticating access key permission to download dataset ...")
    api_url = "https://synapse-response.onrender.com/validate_access"
    user_id = syn.getUserProfile()["ownerId"]
    response = None
    for attempt in range(1, 4):
        try:
            print(f"Requesting access validation, attempt {attempt}/3 ...")
            response = requests.post(
                api_url,
                json={"access_key": access_key, "synapse_id": user_id},
                timeout=180,
            )
            break
        except requests.exceptions.Timeout:
            if attempt == 3:
                print("Access validation timed out after 3 attempts.")
                raise
            print("Access validation timed out; retrying ...")

    if response is None:
        print("Failed to request access: no response received")
        raise SystemExit(1)

    if response.status_code == 200:
        entity_id = response.json()["entity_id"]
    else:
        print("Failed to request access:", response.text)
        raise SystemExit(1)

    # 3. Download dataset to your local folder
    print("Downloading dataset...")
    _ = synapseutils.syncFromSynapse(syn, entity=entity_id, path=str(local_folder))
    print("success!")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with (log_dir / "download_cholectrack20.log").open(
        "a", encoding="utf-8"
    ) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            main()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
