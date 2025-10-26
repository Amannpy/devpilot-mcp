"""
Auth middleware for MCP backend.

Implements JWT token verification for protected routes.
"""

from fastapi import Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError
from typing import Callable

# Secret key for signing tokens (should be in config/env)
SECRET_KEY = "your_super_secret_key"
ALGORITHM = "HS256"


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Verify JWT token in Authorization header for protected endpoints.
        """
        path = request.url.path
        # Skip auth for public endpoints (like /docs or /health)
        if path.startswith("/docs") or path.startswith("/openapi.json") or path.startswith("/health"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

        token = auth_header.split(" ")[1]
        if not self._verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        response = await call_next(request)
        return response

    def _verify_token(self, token: str) -> bool:
        """
        Verify JWT token validity.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            # Optional: attach user info to request.state if needed
            return True
        except JWTError:
            return False
