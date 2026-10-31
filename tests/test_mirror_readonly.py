"""CPU tests of the archive boundary used for read-only evidence collection."""

import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.mirror_readonly import REMOTE_CODE, extract_verified, main


class MirrorArchiveTests(unittest.TestCase):
    def _archive(self, root: Path, name: str = "run/manifest.json",
                 digest: str | None = None, link: bool = False) -> Path:
        data = b'{"status":"fixture"}\n'
        manifest = {"files": [{"relative_path": name, "size": len(data),
                               "sha256": digest or hashlib.sha256(data).hexdigest()}]}
        path = root / "snapshot.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/forbidden"
                info.size = 0
            archive.addfile(info, None if link else io.BytesIO(data))
            blob = json.dumps(manifest).encode()
            info = tarfile.TarInfo("SNAPSHOT.json")
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
        return path

    def test_accepts_hash_verified_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = extract_verified(self._archive(root), root / "out")
            self.assertEqual(len(manifest["files"]), 1)
            self.assertEqual((root / "out/run/manifest.json").read_bytes(),
                             b'{"status":"fixture"}\n')

    def test_rejects_traversal_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_verified(self._archive(root, "../escape"), root / "out")
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "escape").exists())

    def test_rejects_symlink_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_verified(self._archive(root, link=True), root / "out")
            self.assertFalse((root / "out").exists())

    def test_rejects_hash_mismatch_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                extract_verified(self._archive(root, digest="0" * 64), root / "out")
            self.assertFalse((root / "out").exists())

    def test_remote_collector_streams_without_filesystem_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "experiments/aiperf/run/manifest.json"
            source.parent.mkdir(parents=True)
            source.write_text('{"fixture":true}\n')
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            payload = {"base": str(root), "files": [
                {"rel": "run/manifest.json", "size": source.stat().st_size}]}
            result = subprocess.run([sys.executable, "-B", "-"],
                                    input=("payload = " + repr(payload) + "\n" + REMOTE_CODE).encode(),
                                    capture_output=True, check=True)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))
            archive = root / "local-copy.tar.gz"
            archive.write_bytes(result.stdout)
            extract_verified(archive, root / "out")
            self.assertEqual((root / "out/run/manifest.json").read_bytes(), source.read_bytes())

    def test_large_real_raw_exports_require_an_explicit_bounded_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "experiments/aiperf/run/raw.jsonl"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"x" * (17 * 1024 * 1024))
            payload = {"base": str(root), "files": [{"rel": "run/raw.jsonl", "size": source.stat().st_size}]}
            command = [sys.executable, "-B", "-"]
            rejected = subprocess.run(command, input=("payload = " + repr(payload) + "\n" + REMOTE_CODE).encode(), capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            payload["max_file_mib"] = 128
            accepted = subprocess.run(command, input=("payload = " + repr(payload) + "\n" + REMOTE_CODE).encode(), capture_output=True, check=True)
            archive = root / "local.tar.gz"; archive.write_bytes(accepted.stdout)
            with self.assertRaisesRegex(ValueError, "size limit"):
                extract_verified(archive, root / "rejected")
            extract_verified(archive, root / "accepted", 128)
            self.assertEqual((root / "accepted/run/raw.jsonl").read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError): extract_verified(archive, root / "unbounded", 1024)

    def test_timeout_retains_partial_and_writes_failure_journal_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selection = root / "selection.json"
            selection.write_text('{"files": []}')
            out = root / "out"
            def timeout(*args, **kwargs):
                kwargs["stdout"].write(b"partial-fixture")
                raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr=b"fixture timeout")
            argv = ["mirror", "--target", "fixture", "--base", str(root), "--selection", str(selection),
                    "--out", str(out), "--transfer-timeout-seconds", "900"]
            with patch.object(sys, "argv", argv), patch("scripts.mirror_readonly.subprocess.run", side_effect=timeout):
                with self.assertRaisesRegex(SystemExit, "timeout=True"):
                    main()
            journal = json.loads((out / "transport.json").read_text())
            self.assertTrue(journal["timed_out"])
            self.assertIsNone(journal["rc"])
            self.assertEqual(journal["timeout_seconds"], 900)
            self.assertEqual((out / "snapshot.tar.gz").read_bytes(), b"partial-fixture")
            self.assertFalse((out / "files").exists())

    def test_256mib_profile_requires_exact_sealed_raw_and_preserves_old_default_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            relative = 'agent_work/runs/cpu-fixture/output/bf16/artifacts/profile_export_raw.jsonl'
            source = root/relative; source.parent.mkdir(parents=True)
            size = 129 * 1024 * 1024
            with source.open('xb') as stream: stream.truncate(size)
            hasher = hashlib.sha256()
            for _ in range(129): hasher.update(b'\0' * 1024 * 1024)
            run = root/'agent_work/runs/cpu-fixture'
            (run/'ticket.json').write_text(json.dumps({'request': {'run_id': 'cpu-fixture'}, 'request_sha256': 'fixture'}))
            measurement = source.parent.parent/'measurement.json'
            measurement.write_text(json.dumps({'measurement_success': True, 'arm': 'bf16', 'request_sha256': 'fixture',
                'evidence_sha256': {'artifacts/profile_export_raw.jsonl': hasher.hexdigest()}}))
            payload = {'base': str(root), 'max_file_mib': 256,
                       'files': [{'base_rel': relative, 'size': size}]}
            command = [sys.executable, '-B', '-']
            accepted = subprocess.run(command, input=('payload = '+repr(payload)+'\n'+REMOTE_CODE).encode(),
                                      capture_output=True, check=True)
            archive = root/'local.gz'; archive.write_bytes(accepted.stdout)
            with self.assertRaisesRegex(ValueError, 'size limit'):
                extract_verified(archive, root/'old-bound', 128)
            self.assertFalse((root/'old-bound').exists())
            manifest = extract_verified(archive, root/'new-bound', 256)
            self.assertEqual(manifest['files'][0]['sha256'], hasher.hexdigest())
            self.assertEqual((root/'new-bound/support'/relative).stat().st_size, size)
            measurement.write_text(json.dumps({'measurement_success': False}))
            refused = subprocess.run(command, input=('payload = '+repr(payload)+'\n'+REMOTE_CODE).encode(), capture_output=True)
            self.assertNotEqual(refused.returncode, 0)

    def _xz_stream(self, root: Path) -> bytes:
        source = root / 'experiments/aiperf/run/source.json'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'{"fixture":"CPU-only XZ stream"}\n' * 100)
        payload = {'base': str(root.resolve()), 'archive_codec': 'xz',
                   'files': [{'rel': 'run/source.json', 'size': source.stat().st_size}]}
        before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
        result = subprocess.run([sys.executable, '-B', '-'],
            input=('payload = ' + repr(payload) + '\n' + REMOTE_CODE).encode(), capture_output=True, check=True)
        self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))
        return result.stdout

    def test_xz_readonly_stream_and_bounded_local_extraction_preserve_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = root/'snapshot.tar.xz'; archive.write_bytes(self._xz_stream(root))
            result = extract_verified(archive, root/'out', archive_codec='xz')
            self.assertEqual(result['archive_codec'], 'xz')
            self.assertEqual((root/'out/run/source.json').read_bytes(),
                             (root/'experiments/aiperf/run/source.json').read_bytes())

    def test_xz_truncation_trailing_bytes_and_concatenation_never_extract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); encoded = self._xz_stream(root)
            for index, data in enumerate((encoded[:-8], encoded+b'tail', encoded+encoded)):
                archive = root/f'bad{index}.xz'; archive.write_bytes(data)
                with self.assertRaises((ValueError, lzma.LZMAError)):
                    extract_verified(archive, root/f'out{index}', archive_codec='xz')
                self.assertFalse((root/f'out{index}').exists())

    def test_xz_expansion_and_decoder_memory_limits_are_enforced_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); encoded = self._xz_stream(root)
            for name, constant, limit in (('size', 'MAX_EXPANDED_TAR_BYTES', 1000),
                                          ('memory', 'XZ_MEMORY_LIMIT_BYTES', 1024)):
                archive = root/(name+'.xz'); archive.write_bytes(encoded)
                with patch('scripts.mirror_readonly.'+constant, limit), self.assertRaises((ValueError, lzma.LZMAError)):
                    extract_verified(archive, root/(name+'-out'), archive_codec='xz')
                self.assertFalse((root/(name+'-out')).exists())


if __name__ == "__main__":
    unittest.main()
