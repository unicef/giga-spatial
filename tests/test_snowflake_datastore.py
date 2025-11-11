import os
import uuid
import pytest


# Run these integration tests only when explicitly enabled.
# Set GIGASPATIAL_RUN_SNOWFLAKE_TESTS=1 to enable.
RUN_INTEGRATION = os.getenv("GIGASPATIAL_RUN_SNOWFLAKE_TESTS") == "1"

snowflake = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set GIGASPATIAL_RUN_SNOWFLAKE_TESTS=1 to run Snowflake integration tests",
)


@pytest.fixture(scope="module")
def store():
    if not RUN_INTEGRATION:
        pytest.skip("Snowflake integration disabled")

    # Import here to avoid import-time costs when skipped
    from gigaspatial.core.io import SnowflakeDataStore

    instance = SnowflakeDataStore()
    try:
        yield instance
    finally:
        try:
            instance.close()
        except Exception:
            pass


@pytest.fixture(scope="module")
def test_prefix():
    # Unique namespace per test module run
    return f"validation_test_pytest_{uuid.uuid4().hex}"


@snowflake
def test_text_io_and_exists(store, test_prefix):
    test_path = f"{test_prefix}/simple_text.txt"
    content = "Hello from SnowflakeDataStore via pytest!\nLine 2.\n"

    # Write + existence
    store.write_file(test_path, content)
    assert store.file_exists(test_path) is True
    assert store.exists(test_path) is True
    assert store.is_file(test_path) is True
    assert store.is_dir(test_path) is False

    # Read + equality
    read_back = store.read_file(test_path, encoding="utf-8")
    assert read_back == content


@snowflake
def test_binary_io(store, test_prefix):
    path = f"{test_prefix}/binary_file.bin"
    data = b"\x00\x01\x02\x03\xff\xfe pytest"

    store.write_file(path, data)
    assert store.file_exists(path) is True

    read_back = store.read_file(path)
    assert isinstance(read_back, (bytes, bytearray))
    assert bytes(read_back) == data


@snowflake
def test_directories_and_listing(store, test_prefix):
    files = [
        (f"{test_prefix}/dir1/file1.txt", "File 1"),
        (f"{test_prefix}/dir1/file2.txt", "File 2"),
        (f"{test_prefix}/dir2/subdir/file3.txt", "File 3"),
    ]
    for p, c in files:
        store.write_file(p, c)

    # list_files root
    all_files = store.list_files(test_prefix)
    assert isinstance(all_files, list)
    # At least the files we just wrote
    for p, _ in files:
        assert p in all_files

    # dir vs file checks
    assert store.is_dir(f"{test_prefix}/dir1") is True
    assert store.is_dir(f"{test_prefix}/dir2/subdir") is True
    assert store.is_file(f"{test_prefix}/dir1/file1.txt") is True


@snowflake
def test_copy_and_rename_and_remove(store, test_prefix):
    src = f"{test_prefix}/copy_src.txt"
    dst = f"{test_prefix}/copy_dst.txt"
    renamed = f"{test_prefix}/renamed.txt"

    content = "Copy me"
    store.write_file(src, content)

    # copy
    store.copy_file(src, dst)
    assert store.file_exists(dst) is True
    assert store.read_file(dst, encoding="utf-8") == content

    # rename
    store.rename(dst, renamed)
    assert store.file_exists(dst) is False
    assert store.file_exists(renamed) is True
    assert store.read_file(renamed, encoding="utf-8") == content

    # remove
    store.remove(renamed)
    assert store.file_exists(renamed) is False


@snowflake
def test_open_context_manager(store, test_prefix):
    path = f"{test_prefix}/ctx.txt"
    text = "Written via context manager"

    with store.open(path, "w") as f:
        f.write(text)

    with store.open(path, "r") as f:
        read_back = f.read()

    assert read_back == text


@snowflake
def test_file_size_and_metadata(store, test_prefix):
    path = f"{test_prefix}/sized.txt"
    text = "abc" * 10
    store.write_file(path, text)

    size_kb = store.file_size(path)
    assert size_kb >= 0.0

    meta = store.get_file_metadata(path)
    assert isinstance(meta, dict)
    # Common expected keys from notebook
    for key in ("name", "size_bytes"):
        assert key in meta


@snowflake
def test_rmdir_and_mkdir(store, test_prefix):
    base_dir = f"{test_prefix}/mkdir_test"
    subfile = f"{base_dir}/file.txt"

    # mkdir and write
    store.mkdir(base_dir)
    assert store.is_dir(base_dir) is True
    store.write_file(subfile, "payload")
    assert store.file_exists(subfile) is True

    # rmdir should remove contents
    store.rmdir(base_dir)
    # Subsequent listing should be empty or not include the dir
    listed = store.list_files(test_prefix)
    assert all(not p.startswith(base_dir) for p in listed)


@snowflake
def test_cleanup_namespace(store, test_prefix):
    # Final cleanup of the whole namespace
    try:
        store.rmdir(test_prefix)
    except Exception:
        # If not empty or already removed, ignore to not fail the suite end
        pass
