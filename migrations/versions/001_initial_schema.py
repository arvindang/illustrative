"""Initial schema with users and graphic_novels tables.

Revision ID: 001
Revises:
Create Date: 2026-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('gemini_api_key_encrypted', sa.LargeBinary(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('idx_users_email', 'users', ['email'])

    # Create graphic_novels table
    op.create_table(
        'graphic_novels',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('source_filename', sa.String(255), nullable=True),
        sa.Column('art_style', sa.String(100), nullable=True),
        sa.Column('narrative_tone', sa.String(100), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(50), server_default='processing'),
        sa.Column('pdf_storage_key', sa.String(512), nullable=True),
        sa.Column('epub_storage_key', sa.String(512), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('idx_novels_user_id', 'graphic_novels', ['user_id'])
    op.create_index('idx_novels_status', 'graphic_novels', ['status'])


def downgrade() -> None:
    op.drop_index('idx_novels_status', table_name='graphic_novels')
    op.drop_index('idx_novels_user_id', table_name='graphic_novels')
    op.drop_table('graphic_novels')
    op.drop_index('idx_users_email', table_name='users')
    op.drop_table('users')
