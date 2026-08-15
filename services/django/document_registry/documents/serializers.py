from rest_framework import serializers

from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "source_uri", "title", "tags", "reviewed", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]
