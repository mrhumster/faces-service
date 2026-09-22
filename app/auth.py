import time

import httpx
import jwt

from . import config


class AuthError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


class TokenService:
    def __init__(self) -> None:
        self._public_key_pem = self._fetch_public_key()

    @staticmethod
    def _fetch_public_key() -> str:
        url = config.Config.jwt_public_key_url
        if not url:
            raise AuthError(500, "JWT_ACCESS_PUBLIC_KEY_URL not set")
        r = httpx.get(url, timeout=10)
        if r.status_code != 200:
            raise AuthError(500, f"public key fetch failed: {r.status_code}")
        return r.text

    def validate(self, token: str) -> dict:
        try:
            claims = jwt.decode(
                token,
                self._public_key_pem,
                algorithms=["RS256"],
                options={"verify_exp": True, "verify_aud": False},
            )
        except jwt.ExpiredSignatureError:
            raise AuthError(401, "token expired")
        except jwt.InvalidTokenError:
            raise AuthError(401, "invalid token")
        return claims

    def refresh(self) -> None:
        """Re-fetch the public key (rotated keys)."""
        self._public_key_pem = self._fetch_public_key()


class AuthContext:
    def __init__(self, claims: dict, is_internal: bool = False) -> None:
        self.user_id: str | None = claims.get("user_id") if claims else None
        self.role: str | None = claims.get("role") if claims else None
        self.email_verified: bool = bool(claims.get("email_verified")) if claims else False
        self.is_internal = is_internal


_token_service: TokenService | None = None


def init_token_service() -> TokenService:
    global _token_service
    if _token_service is None:
        _token_service = TokenService()
    return _token_service


def get_token_service() -> TokenService:
    if _token_service is None:
        raise AuthError(503, "token service not initialized")
    return _token_service


def authorized(auth_header: str | None) -> AuthContext:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise AuthError(401, "missing bearer token")
    token = auth_header[len("Bearer ") :].strip()
    if not token:
        raise AuthError(401, "missing bearer token")
    try:
        svc = get_token_service()
    except AuthError:
        raise AuthError(503, "token service unavailable")
    try:
        claims = svc.validate(token)
    except AuthError:
        raise
    except Exception:
        raise AuthError(401, "invalid token")
    if not claims.get("user_id"):
        raise AuthError(401, "token missing user_id")
    return AuthContext(claims, is_internal=False)


def internal_token(token: str | None) -> None:
    if not config.Config.internal_token:
        raise AuthError(503, "internal token not configured")
    if token != config.Config.internal_token:
        raise AuthError(401, "invalid internal token")


def ensure_owner(auth: AuthContext, owner_id: str) -> None:
    """Owner or admin may access."""
    if auth.role == "admin":
        return
    if auth.user_id != owner_id:
        raise AuthError(404, "not found")