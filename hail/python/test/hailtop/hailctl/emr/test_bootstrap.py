import subprocess
from importlib import resources


def test_bootstrap_rejects_invalid_wheel_filename_before_installing():
    script = resources.files('hailtop.hailctl.emr').joinpath('resources/install-hail-emr.sh')
    result = subprocess.run(
        ['bash', str(script), 's3://bucket/hail.whl', 'a' * 64, 's3://bucket/wheelhouse.tar.gz', 'b' * 64],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'does not end in a valid Hail wheel filename' in result.stderr
