import pytest


@pytest.mark.asyncio
async def test_db_fixture_is_live(db):
    """db fixture must yield a real migrated batch database."""
    rows = await db.execute_and_fetchall('SHOW TABLES')
    table_names = {next(iter(row.values())) for row in rows}
    assert 'billing_projects' in table_names
