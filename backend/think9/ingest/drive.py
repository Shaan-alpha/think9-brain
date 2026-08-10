"""Google Drive connector, plus the local-mirror fallback of spec section 12.

Both clients expose the same two methods, so the ingest pipeline cannot tell which one it
is holding. That is the point: if the service-account setup stalls, the code path under
test stays identical and only the source changes.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_DOC = "application/vnd.google-apps.document"
GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
_FIELDS = "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)"


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    modified_time: str
    web_view_link: str


def build_service(credentials_json: str):
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


class DriveClient:
    def __init__(self, service) -> None:
        self.service = service

    def list_folder(self, folder_id: str) -> list[DriveFile]:
        files: list[DriveFile] = []
        page_token: str | None = None
        while True:
            kwargs = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": _FIELDS,
                "pageSize": 100,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = self.service.files().list(**kwargs).execute()
            files.extend(
                DriveFile(f["id"], f["name"], f["mimeType"], f["modifiedTime"], f["webViewLink"])
                for f in response.get("files", [])
            )
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def fetch(self, file: DriveFile) -> bytes:
        if file.mime_type in (GOOGLE_DOC, GOOGLE_SHEET):
            mime = "text/csv" if file.mime_type == GOOGLE_SHEET else "text/plain"
            return self.service.files().export_media(fileId=file.id, mimeType=mime).execute()
        return self.service.files().get_media(fileId=file.id).execute()


class LocalFolderClient:
    """Reads the generated corpus mirror instead of Drive.

    Used when GOOGLE_CREDENTIALS_JSON is unset. The README says so plainly rather than
    implying live ingestion that is not happening.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # folder_id is unused here but kept for interface parity with DriveClient.
    def list_folder(self, folder_id: str) -> list[DriveFile]:
        return [
            DriveFile(
                id=hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:24],
                name=path.name,
                mime_type="text/markdown",
                modified_time=str(path.stat().st_mtime),
                web_view_link=path.resolve().as_uri(),
            )
            for path in sorted(self.root.glob("*.md"))
        ]

    def fetch(self, file: DriveFile) -> bytes:
        return (self.root / file.name).read_bytes()
