from fastapi import APIRouter

from app.shared.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "UP",
        "service": settings.service_name,
    }
