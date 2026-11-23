"""WSGI entry point for gunicorn (Docker Compose)."""

from app import create_app

application = create_app()
