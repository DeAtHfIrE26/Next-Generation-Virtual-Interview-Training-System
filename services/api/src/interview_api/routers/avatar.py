"""Optional neural talking-head (product feature, not a patent element).

The GPU service is external (see services/gpu/README.md for the contract) and must use a
commercially licensed model. The client gives it a hard time budget and falls back to the
in-browser viseme rig on any error, timeout or when this feature is off.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from interview_api import metering
from interview_api.db import get_db
from interview_api.models import User
from interview_api.ratelimit import limiter
from interview_api.security import current_user
from interview_api.settings import get_settings

router = APIRouter(prefix="/avatar", tags=["avatar"])
render_limit = limiter("avatar", capacity=30, per_seconds=60)


class RenderIn(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    session_id: str | None = Field(default=None, max_length=32)


@router.post("/render", dependencies=[Depends(render_limit)])
def render(body: RenderIn, user: User = Depends(current_user), db=Depends(get_db)) -> dict:
    s = get_settings()
    if not (s.feature_neural_avatar and s.neural_avatar_url):
        raise HTTPException(501, "neural avatar is not enabled")
    try:
        r = httpx.post(f"{s.neural_avatar_url.rstrip('/')}/render", json={"text": body.text}, timeout=1.5)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(503, "avatar service unavailable") from e
    metering.record(
        db, user, body.session_id, "gpu_seconds", "neural_avatar", float(data.get("gpu_seconds", 0))
    )
    db.commit()
    return {"stream_url": data["stream_url"]}
