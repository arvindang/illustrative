"""Add progress tracking columns to graphic_novels table.

Revision ID: 003
Revises: 002
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Progress tracking columns
    op.add_column('graphic_novels', sa.Column('current_stage', sa.String(50), nullable=True))
    op.add_column('graphic_novels', sa.Column('pages_completed', sa.Integer(), server_default='0'))
    op.add_column('graphic_novels', sa.Column('pages_total', sa.Integer(), nullable=True))
    op.add_column('graphic_novels', sa.Column('panels_completed', sa.Integer(), server_default='0'))
    op.add_column('graphic_novels', sa.Column('panels_total', sa.Integer(), nullable=True))

    # Resume capability columns
    op.add_column('graphic_novels', sa.Column('can_resume', sa.Boolean(), server_default='false'))
    op.add_column('graphic_novels', sa.Column('manifest_path', sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column('graphic_novels', 'manifest_path')
    op.drop_column('graphic_novels', 'can_resume')
    op.drop_column('graphic_novels', 'panels_total')
    op.drop_column('graphic_novels', 'panels_completed')
    op.drop_column('graphic_novels', 'pages_total')
    op.drop_column('graphic_novels', 'pages_completed')
    op.drop_column('graphic_novels', 'current_stage')
