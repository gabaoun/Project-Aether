# Document Registry (Django REST Framework)

Complementary microservice to the main FastAPI app - a tagging/review
registry for documents ingested into Project Aether's Chroma Cloud index
(`../../../src/pipeline/ingestion.py`). Owns its own table (`documents_document`)
in the shared Postgres instance; does not touch the main app's
Alembic-managed schema.

Follows this repo's `services/` convention (alongside `services/go/*`) for
polyglot microservices that complement, rather than replace, the core Python
FastAPI app.

## Stack

Django 6.1 + Django REST Framework, token-authenticated CRUD, PostgreSQL
(SQLite fallback for local dev/tests when `DJANGO_DB_NAME` isn't set - see
`config/settings.py`).

## Endpoints

| Method | Route                  | Auth              | Description        |
| :----- | :---------------------- | :----------------- | :------------------ |
| POST   | `/api-token-auth/`      | username + password | Obtain a DRF token |
| GET    | `/api/documents/`       | Token              | List documents      |
| POST   | `/api/documents/`       | Token              | Create a document   |
| GET    | `/api/documents/{id}/`  | Token              | Retrieve a document |
| PATCH  | `/api/documents/{id}/`  | Token              | Update (e.g. mark `reviewed`) |
| DELETE | `/api/documents/{id}/`  | Token              | Delete a document   |

## Running locally

```bash
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser  # to get a user for /api-token-auth/
python manage.py runserver
```

Uses local SQLite by default. To point at the shared Postgres instance
(`../../../docker-compose.yml`'s `postgres` service), set:

```bash
export DJANGO_DB_NAME=aether_documents
export DJANGO_DB_USER=user
export DJANGO_DB_PASSWORD=password
export DJANGO_DB_HOST=postgres
export DJANGO_DB_PORT=5432
```

## Tests

```bash
python manage.py test
```

6 tests covering: auth-required rejection, create, list, update (review
flag), delete, and the `source_uri` uniqueness constraint.
