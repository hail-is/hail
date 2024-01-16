import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from jproperties import Properties

from hailtop.config.user_config import configuration_of
from hailtop.config.variables import ConfigVariable

GCSRequesterPaysConfiguration = Union[str, Tuple[str, List[str]]]


def get_gcs_requester_pays_configuration(
    *,
    gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
) -> Optional[GCSRequesterPaysConfiguration]:
    if gcs_requester_pays_configuration:
        return gcs_requester_pays_configuration

    project = configuration_of(ConfigVariable.GCS_REQUESTER_PAYS_PROJECT, None, None)
    buckets = configuration_of(ConfigVariable.GCS_REQUESTER_PAYS_BUCKETS, None, None)

    spark_conf = get_spark_conf_gcs_requester_pays_configuration()

    if spark_conf and (project or buckets):
        warnings.warn(
            f'You have specified the GCS requester pays configuration in both your '
            f'spark-defaults.conf ({spark_conf.conf_path}) and either an explicit argument or '
            f'through `hailctl config`. For GCS requester pays configuration, Hail '
            'first checks explicit arguments, then `hailctl config`, then '
            'spark-defaults.conf.'
        )

    if project is not None:
        if buckets is not None:
            return (project, buckets.split(','))
        return project

    if spark_conf is None:
        return None
    warnings.warn(
        'Reading spark-defaults.conf to determine GCS requester pays '
        'configuration. This is deprecated. Please use `hailctl config '
        'set gcs_requester_pays/project` and `hailctl config set '
        'gcs_requester_pays/buckets`.'
    )

    # https://github.com/GoogleCloudDataproc/hadoop-connectors/blob/master/gcs/CONFIGURATION.md#cloud-storage-requester-pays-feature-configuration
    if spark_conf.mode is None:
        if spark_conf.project or spark_conf.buckets:
            warnings.warn(
                f'When reading GCS requester pays configuration from spark-defaults.conf '
                f'({spark_conf.conf_path}), no mode is set, so requester pays '
                f'will be disabled.'
            )
        return None

    if spark_conf.mode == SparkConfGcsRequesterPaysMode.DISABLED:
        return None

    if spark_conf.project is None:
        raise ValueError(
            f'When reading GCS requester pays configuration from spark-defaults.conf '
            f'({spark_conf.conf_path}), a project must be set if a mode other than '
            f'DISABLED is set.'
        )

    if spark_conf.mode == SparkConfGcsRequesterPaysMode.CUSTOM:
        if spark_conf.buckets is None:
            raise ValueError(
                f'When reading GCS requester pays configuration from '
                f'spark-defaults.conf ({spark_conf.conf_path}) with mode CUSTOM, '
                f'buckets must be set.'
            )
        return (spark_conf.project, spark_conf.buckets)

    if spark_conf.mode == SparkConfGcsRequesterPaysMode.ENABLED:
        warnings.warn(
            f'When reading GCS requester pays configuration from '
            f'spark-defaults.conf ({spark_conf.conf_path}) Hail treats the mode '
            f'ENABLED as AUTO.'
        )

    if spark_conf.buckets is not None:
        warnings.warn(
            f'When reading GCS requester pays configuration from '
            f'spark-defaults.conf ({spark_conf.conf_path}) with mode '
            f'{spark_conf.mode}, found buckets: {spark_conf.buckets}.'
            f'The buckets are ignored in this mode.'
        )

    return spark_conf.project


class SparkConfGcsRequesterPaysMode(Enum):
    ENABLED = 'ENABLED'
    DISABLED = 'DISABLED'
    AUTO = 'AUTO'
    CUSTOM = 'CUSTOM'


@dataclass
class SparkConfGcsRequseterPaysConfiguration:
    mode: Optional[SparkConfGcsRequesterPaysMode]
    project: Optional[str]
    buckets: Optional[List[str]]
    conf_path: str


def spark_conf_path() -> Optional[str]:
    try:
        from pyspark.find_spark_home import _find_spark_home  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return os.path.join(_find_spark_home(), 'conf', 'spark-defaults.conf')


def get_spark_conf_gcs_requester_pays_configuration() -> Optional[SparkConfGcsRequseterPaysConfiguration]:
    mode: Optional[SparkConfGcsRequesterPaysMode] = None
    project: Optional[str] = None
    buckets: Optional[List[str]] = None
    path = spark_conf_path()
    if path is not None and os.path.exists(path):
        props = Properties()
        with open(path, 'rb') as f:
            props.load(f, 'utf-8')
        for key, (val, _) in dict(props).items():
            if key == 'spark.hadoop.fs.gs.requester.pays.mode':
                try:
                    mode = SparkConfGcsRequesterPaysMode(val)
                except ValueError as exc:
                    raise ValueError(
                        f'When reading GCS requester pays configuration from '
                        f'spark-defaults.conf ({path}) an unknown mode was '
                        f'found: {val}. Expected ENABLED, AUTO, CUSTOM, or '
                        f'DISABLED.'
                    ) from exc
            elif key == 'spark.hadoop.fs.gs.requester.pays.project.id':
                project = val
            elif key == 'spark.hadoop.fs.gs.requester.pays.buckets':
                buckets = val.split(',')
        if mode or project or buckets:
            return SparkConfGcsRequseterPaysConfiguration(mode, project, buckets, path)
    return None
