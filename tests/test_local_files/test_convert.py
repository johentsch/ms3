import logging
import subprocess

from ms3.utils import functions


def test_convert_folder_iterative(monkeypatch, tmp_path):
    """The non-parallel branch used to fail unpacking the conversion parameters."""
    (tmp_path / "piece.mscx").write_text("")
    target_dir = tmp_path / "converted"
    calls = []
    monkeypatch.setattr(functions, "get_musescore", lambda ms, logger=None: ms)
    monkeypatch.setattr(functions, "convert", lambda **kwargs: calls.append(kwargs))
    functions.convert_folder(
        directory=str(tmp_path),
        target_dir=str(target_dir),
        target_extension="mei",
        ms="mscore",
        parallel=False,
        logger="ms3.convert",
    )
    assert calls == [
        dict(
            old=str(tmp_path / "piece.mscx"),
            new=str(target_dir / "piece.mei"),
            MS="mscore",
            # a name rather than a Logger, which would lose its custom makeRecord when sent to a worker process
            logger="ms3.convert",
        )
    ]


def test_failed_ms4_conversion_names_the_file(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        functions, "get_major_version_of_musescore_executable", lambda ms: 4
    )
    monkeypatch.setattr(
        functions.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, returncode=1, stdout="", stderr="crashed"
        ),
    )
    old = str(tmp_path / "piece.mscx")
    logger = logging.getLogger("test_convert")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        functions.convert(old, str(tmp_path / "piece.mei"), MS="mscore", logger=logger)
    assert old in caplog.text
    assert "crashed" in caplog.text
