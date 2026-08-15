from django.db import models


class Document(models.Model):
    """
    Metadata/review registry for documents ingested into Project Aether's
    Chroma Cloud index (src/pipeline/ingestion.py). Deliberately owns its
    own table in the shared Postgres instance rather than touching the main
    app's Alembic-managed `ingestion_jobs` table - this is a complementary
    read/write surface (tagging, review status), not a duplicate of it.
    """

    source_uri = models.CharField(max_length=2048, unique=True)
    title = models.CharField(max_length=255)
    tags = models.JSONField(default=list, blank=True)
    reviewed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title
