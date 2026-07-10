"""Username resolution endpoint."""

from fastapi import APIRouter

from ...schemas.users import UserResolveRequest, UserResponse
from ...services import UserService
from ..dependencies import DatabaseSession

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "/resolve",
    response_model=UserResponse,
    summary="Resolve or create a logical username",
)
async def resolve_user(request: UserResolveRequest, session: DatabaseSession) -> UserResponse:
    service = UserService(session)
    user, created = service.resolve(request.username)
    return service.response(user, created)
