from typing import ClassVar

from rest_framework.authentication import BaseAuthentication, TokenAuthentication
from rest_framework.permissions import BasePermission, IsAuthenticated
from rest_framework.viewsets import ModelViewSet

from .models import Document
from .serializers import DocumentSerializer


class DocumentViewSet(ModelViewSet):
    """
    CRUD over the document registry. Token-authenticated - see
    rest_framework.authtoken (DRF's built-in TokenAuthentication), issued
    via the standard /api-token-auth/ endpoint wired in config/urls.py.
    """

    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    authentication_classes: ClassVar[list[type[BaseAuthentication]]] = [TokenAuthentication]
    permission_classes: ClassVar[list[type[BasePermission]]] = [IsAuthenticated]
