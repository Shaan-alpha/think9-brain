from think9.ingest.drive import DriveClient, DriveFile, LocalFolderClient


class FakeExecutable:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeFiles:
    def __init__(self, listing):
        self.listing = listing
        self.calls: list[dict] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return FakeExecutable(self.listing)

    def export_media(self, fileId, mimeType):  # Google's camelCase parameter names
        self.calls.append({"export": fileId, "mimeType": mimeType})
        return FakeExecutable(b"exported text")

    def get_media(self, fileId):  # Google's camelCase parameter name
        self.calls.append({"get_media": fileId})
        return FakeExecutable(b"%PDF-1.4 binary")


class FakeService:
    def __init__(self, listing):
        self._files = FakeFiles(listing)

    def files(self):
        return self._files


def _raw(fid, mime="text/plain"):
    return {
        "id": fid,
        "name": fid,
        "mimeType": mime,
        "modifiedTime": "2026-01-08T10:00:00.000Z",
        "webViewLink": f"https://drive/{fid}",
    }


def test_list_folder_filters_on_parent_and_excludes_trashed():
    service = FakeService({"files": [_raw("f1", "application/vnd.google-apps.document")]})
    client = DriveClient(service)

    files = client.list_folder("folder-abc")

    assert files == [
        DriveFile(
            "f1",
            "f1",
            "application/vnd.google-apps.document",
            "2026-01-08T10:00:00.000Z",
            "https://drive/f1",
        )
    ]
    query = service.files().calls[0]["q"]
    assert "'folder-abc' in parents" in query
    assert "trashed = false" in query


def test_google_docs_are_exported_as_plain_text():
    service = FakeService({"files": []})
    client = DriveClient(service)
    doc = DriveFile("f1", "memo", "application/vnd.google-apps.document", "t", "link")

    assert client.fetch(doc) == b"exported text"
    assert service.files().calls[-1]["mimeType"] == "text/plain"


def test_binary_files_are_downloaded_not_exported():
    service = FakeService({"files": []})
    client = DriveClient(service)
    pdf = DriveFile("f2", "contract", "application/pdf", "t", "link")

    assert client.fetch(pdf) == b"%PDF-1.4 binary"
    assert "get_media" in service.files().calls[-1]


def test_pagination_follows_the_next_page_token():
    class Paged(FakeFiles):
        def list(self, **kwargs):
            self.calls.append(kwargs)
            if "pageToken" not in kwargs:
                return FakeExecutable({"files": [_raw("f1")], "nextPageToken": "page-2"})
            return FakeExecutable({"files": [_raw("f2")]})

    service = FakeService({"files": []})
    service._files = Paged({"files": []})
    client = DriveClient(service)

    assert [f.id for f in client.list_folder("folder-abc")] == ["f1", "f2"]


# --- The section 12 fallback -------------------------------------------------
# Identical interface, different source. The ingest pipeline cannot tell them apart.


def test_local_folder_client_lists_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not markdown", encoding="utf-8")

    files = LocalFolderClient(tmp_path).list_folder("unused")

    assert sorted(f.name for f in files) == ["a.md", "b.md"]


def test_local_folder_client_fetches_bytes(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## S\nbody\n", encoding="utf-8")
    client = LocalFolderClient(tmp_path)

    file = client.list_folder("unused")[0]

    assert b"## S" in client.fetch(file)


def test_local_folder_client_ids_are_stable_across_calls(tmp_path):
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    client = LocalFolderClient(tmp_path)

    first = client.list_folder("unused")[0]
    second = client.list_folder("unused")[0]

    assert first.id == second.id


def test_local_folder_client_deep_link_points_at_the_file(tmp_path):
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")

    file = LocalFolderClient(tmp_path).list_folder("unused")[0]

    assert file.web_view_link.startswith("file://")
    assert "a.md" in file.web_view_link
