import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r'[0-9a-f]{64}')
_GIT_SHA_RE = re.compile(r'[0-9a-f]{40}')
_VERSION_RE = re.compile(r'\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9.-]+)?')


@dataclass(frozen=True)
class HailArtifactManifest:
    schema_version: int
    hail_git_revision: str
    hail_pip_version: str
    spark_version: str
    scala_version: str
    python_version: str
    wheel_uri: str
    wheel_sha256: str
    wheelhouse_uri: str
    wheelhouse_sha256: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'HailArtifactManifest':
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = required - data.keys()
        unknown = data.keys() - required
        if missing:
            raise ValueError(f"artifact manifest is missing field(s): {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"artifact manifest has unknown field(s): {', '.join(sorted(unknown))}")
        manifest = cls(**data)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f'unsupported artifact manifest schema_version: {self.schema_version}')
        if _GIT_SHA_RE.fullmatch(self.hail_git_revision) is None:
            raise ValueError('hail_git_revision must be a lowercase 40-character git SHA')
        if not self.hail_pip_version:
            raise ValueError('hail_pip_version must not be empty')
        for name, version in (
            ('spark_version', self.spark_version),
            ('scala_version', self.scala_version),
            ('python_version', self.python_version),
        ):
            if _VERSION_RE.fullmatch(version) is None:
                raise ValueError(f'{name} is not a supported version string: {version!r}')
        for name, digest in (
            ('wheel_sha256', self.wheel_sha256),
            ('wheelhouse_sha256', self.wheelhouse_sha256),
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f'{name} must be a lowercase 64-character SHA256 digest')
        for name, uri in (('wheel_uri', self.wheel_uri), ('wheelhouse_uri', self.wheelhouse_uri)):
            if not uri.startswith('s3://'):
                raise ValueError(f'{name} must be an s3:// URI')

    def bootstrap_args(self) -> list[str]:
        return [
            self.wheel_uri,
            self.wheel_sha256,
            self.wheelhouse_uri,
            self.wheelhouse_sha256,
        ]


def load_artifact_manifest(path: str) -> HailArtifactManifest:
    manifest_path = Path(path).expanduser()
    try:
        data = json.loads(manifest_path.read_text())
    except OSError as exc:
        raise ValueError(f'could not read artifact manifest {path!r}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'artifact manifest {path!r} is not valid JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('artifact manifest must contain a JSON object')
    return HailArtifactManifest.from_dict(data)


def major_minor(version: str) -> str:
    return '.'.join(version.split('.')[:2])
