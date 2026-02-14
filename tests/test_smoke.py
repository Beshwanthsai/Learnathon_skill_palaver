import subprocess
import sys
from pathlib import Path


def test_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data.csv"
    outdir = tmp_path / "artifacts"
    # generate data
    subprocess.check_call([sys.executable, str(root / "data" / "generate_synthetic.py"), "--output", str(data), "--n", "200"])
    # train
    subprocess.check_call([sys.executable, str(root / "src" / "modeling" / "train.py"), "--data", str(data), "--outdir", str(outdir)])
    assert (outdir / "model.joblib").exists()
    assert (outdir / "metrics.csv").exists()
