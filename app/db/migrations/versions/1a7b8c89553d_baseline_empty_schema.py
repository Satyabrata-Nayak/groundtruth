"""baseline empty schema

Revision ID: 1a7b8c89553d
Revises: 
Create Date: 2026-08-26 23:30:29.639864

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = '1a7b8c89553d'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
