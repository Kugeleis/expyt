"""FastAPI application factory."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="ExpYT — Experiment Evaluation Wizard",
        description="Multi-step wizard for statistical experiment evaluation",
        version="0.1.0",
    )
    return application


app = create_app()
