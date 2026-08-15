from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from .models import Document


class DocumentApiTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="pass1234")
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")

    def test_unauthenticated_request_is_rejected(self):
        self.client.credentials()  # drop the Authorization header
        response = self.client.get("/api/documents/")
        self.assertEqual(response.status_code, 401)

    def test_create_document(self):
        response = self.client.post(
            "/api/documents/",
            {"source_uri": "s3://aether-docs/report.pdf", "title": "Q3 Report", "tags": ["finance"]},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Document.objects.count(), 1)
        self.assertFalse(Document.objects.first().reviewed)

    def test_list_documents(self):
        Document.objects.create(source_uri="s3://a", title="A")
        Document.objects.create(source_uri="s3://b", title="B")

        response = self.client.get("/api/documents/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_update_document_marks_reviewed(self):
        doc = Document.objects.create(source_uri="s3://c", title="C")

        response = self.client.patch(f"/api/documents/{doc.id}/", {"reviewed": True}, format="json")

        self.assertEqual(response.status_code, 200)
        doc.refresh_from_db()
        self.assertTrue(doc.reviewed)

    def test_delete_document(self):
        doc = Document.objects.create(source_uri="s3://d", title="D")

        response = self.client.delete(f"/api/documents/{doc.id}/")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(Document.objects.count(), 0)

    def test_source_uri_must_be_unique(self):
        Document.objects.create(source_uri="s3://dup", title="First")

        response = self.client.post(
            "/api/documents/", {"source_uri": "s3://dup", "title": "Second"}, format="json"
        )

        self.assertEqual(response.status_code, 400)
