import subprocess
import sys
import textwrap


def test_open_reads_back_what_write_file_wrote(tmp_path):
    """A file written through the store must read back through the store.

    write_file() encodes UTF-8, so open() has to decode UTF-8. Left implicit it
    decodes in the locale encoding instead, which round-trips only where that
    happens to be UTF-8.

    Run under -X warn_default_encoding (PEP 597) so the implicit encoding is
    visible on a UTF-8 machine too; comparing the text alone would pass on a
    UTF-8 runner whether or not the bug is present. The warning is collected
    around the single call and filtered to this module, because third-party
    packages also open files without an encoding and would otherwise decide
    the result.
    """
    script = textwrap.dedent(
        f"""
        import warnings
        from gigaspatial.core.io.local_data_store import LocalDataStore

        store = LocalDataStore({str(tmp_path)!r})
        text = "École élémentaire Ouagadougou / 學校"
        store.write_file("school.txt", text)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_back = store.open("school.txt").read()

        assert read_back == text, f"round-trip altered the text: {{read_back!r}}"
        implicit = [
            w for w in caught
            if issubclass(w.category, EncodingWarning)
            and "local_data_store" in w.filename
        ]
        assert not implicit, f"open() fell back to the locale encoding: {{implicit[0].filename}}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", script],
        capture_output=True, timeout=300,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
