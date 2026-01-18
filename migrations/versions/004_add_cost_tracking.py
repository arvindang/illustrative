"""Add cost tracking columns to graphic_novels table.

Revision ID: 004
Revises: 003
Create Date: 2026-01-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Cost tracking columns
    op.add_column('graphic_novels', sa.Column('estimated_cost', sa.Float(), nullable=True))
    op.add_column('graphic_novels', sa.Column('actual_cost', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('graphic_novels', 'actual_cost')
    op.drop_column('graphic_novels', 'estimated_cost')
