from hailtop.batch_client.parse import parse_memory_in_bytes
from batch.utils import adjust_cores_for_packability


def test_packability():
    assert adjust_cores_for_packability(0) == 250
    assert adjust_cores_for_packability(200) == 250
    assert adjust_cores_for_packability(250) == 250
    assert adjust_cores_for_packability(251) == 500
    assert adjust_cores_for_packability(500) == 500
    assert adjust_cores_for_packability(501) == 1000
    assert adjust_cores_for_packability(1000) == 1000
    assert adjust_cores_for_packability(1001) == 2000
    assert adjust_cores_for_packability(2000) == 2000
    assert adjust_cores_for_packability(2001) == 4000
    assert adjust_cores_for_packability(3000) == 4000
    assert adjust_cores_for_packability(4000) == 4000
    assert adjust_cores_for_packability(4001) == 8000
    assert adjust_cores_for_packability(8001) == 16000


def test_memory_str_to_bytes():
    assert parse_memory_in_bytes('7') == 7
    assert parse_memory_in_bytes('1K') == 1000
    assert parse_memory_in_bytes('1Ki') == 1024
