from typing import ClassVar

from rest_framework import serializers

from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields: ClassVar[list[str]] = [
            "id",
            "source_uri",
            "title",
            "tags",
            "reviewed",
            "created_at",
            "updated_at",
        ]
        read_only_fields: ClassVar[list[str]] = ["id", "created_at", "updated_at"]
