import pytest
import hail as hl
from hailtop.hailctl.dev.query import cli


def test_simple_table():
    t = hl.utils.range_table(50, 3)
    t = t.filter((t.idx % 3 == 0) | ((t.idx / 7) % 3 == 0))
    n = t.count()
    print(f'n {n}')
    assert n == 17

@pytest.mark.asyncio
async def test_flags():
    async with cli.QueryClient() as client:
        all_flags, invalid = await client.get_flag([])
        assert len(invalid) == 0

        for test_flag, test_value in all_flags.items():
            old = await client.set_flag(test_flag, "new_value")
            assert old == test_value

            flags, _ = await client.get_flag([test_flag])
            assert flags == {test_flag: "new_value"}

            old = await client.set_flag(test_flag, None)
            assert old == "new_value"

            if test_value is not None:
                old = await client.set_flag(test_flag, test_value)
                assert old is None

        flags, invalid = await client.get_flag(["invalid"])
        assert invalid == ["'invalid'"]
        assert flags == {}
