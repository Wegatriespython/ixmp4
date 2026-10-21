# type: ignore
"""add run.lock_token column

Revision ID: a3f1c5d7e9b2
Revises: ad8eabea98b9
Create Date: 2026-10-13 16:50:00.000000

"""

import logging

import sqlalchemy as sa
from alembic import op

from ixmp4.data.versions import PostgresVersionTriggers

# Revision identifiers, used by Alembic.
revision = "a3f1c5d7e9b2"
down_revision = "ad8eabea98b9"
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)


def _sync_run_version_triggers() -> None:
    conn = op.get_bind()
    if conn is None:
        logger.warning("Cannot sync version triggers without database connection.")
        return

    dialect_name = conn.dialect.name
    if dialect_name != "postgresql":
        logger.info(
            f"Skipping version trigger sync for database dialect '{dialect_name}'."
        )
        return

    metadata = sa.MetaData()
    transaction_table = sa.Table("transaction", metadata, autoload_with=conn)
    data_table = sa.Table("run", metadata, autoload_with=conn)
    version_table = sa.Table("run_version", metadata, autoload_with=conn)
    PostgresVersionTriggers(data_table, version_table, transaction_table).sync_entities(
        conn
    )


def upgrade():
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("lock_token", sa.String(length=32), nullable=True)
        )

    with op.batch_alter_table("run_version", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("lock_token", sa.String(length=32), nullable=True)
        )

    _sync_run_version_triggers()


def downgrade():
    with op.batch_alter_table("run_version", schema=None) as batch_op:
        batch_op.drop_column("lock_token")

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.drop_column("lock_token")

    _sync_run_version_triggers()
