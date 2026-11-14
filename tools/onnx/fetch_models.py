"""Fetch the pinned 512-preset inputs and record verified SHA-256 hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

GGUF_REVISION = "a57397bd3d351599d9729fc144b3f87c3f87d65b"
MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("models"))
    args = parser.parse_args()
    jobs = []
    for repo, revision, names, directory in [
        ("ilintar/trellis2-gguf", GGUF_REVISION,
         [f"{name}.gguf" for name in ["birefnet", "dinov3", "ss_flow", "ss_dec",
                                     "shape_flow_512", "shape_dec", "tex_flow_512", "tex_dec"]],
         args.dest / "f16"),
        ("microsoft/TRELLIS.2-4B", MODEL_REVISION,
         ["pipeline.json", "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
          "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"], args.dest / "reference"),
    ]:
        metadata = {f.rfilename: f for f in HfApi().model_info(repo, revision=revision, files_metadata=True).siblings}
        for name in names:
            jobs.append((repo, revision, name, directory, metadata[name]))

    def download(job):
        repo, revision, name, directory, metadata = job
        print(f"fetch {repo}@{revision[:12]}/{name}", flush=True)
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=directory))
        actual = sha256(path)
        if metadata.lfs and actual != metadata.lfs.sha256:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        print(f"verified {name}: {actual}", flush=True)
        return dict(repo=repo, revision=revision, file=name, path=str(path.resolve()),
                    bytes=path.stat().st_size, sha256=actual)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, jobs))
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
