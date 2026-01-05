"""Add error_message column to graphic_novels table.

Revision ID: 002
Revises: 001
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('graphic_novels', sa.Column('error_message', sa.String(2000), nullable=True))


def downgrade() -> None:
    op.drop_column('graphic_novels', 'error_message')
