from hailtop.hailctl.emr import submit


def test_spark_submit_step_args():
    args = submit.spark_submit_step_args('s3://bkt/scripts/x.py', ['--foo', 'bar'])
    assert args[0] == 'spark-submit'
    assert args[-3:] == ['s3://bkt/scripts/x.py', '--foo', 'bar']


def test_spark_submit_step_args_no_passthrough():
    args = submit.spark_submit_step_args('s3://bkt/scripts/x.py', [])
    assert args == ['spark-submit', '--deploy-mode', 'client', 's3://bkt/scripts/x.py']
