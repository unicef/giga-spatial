from enum import Enum
from typing import Any, Optional, Tuple

from pydantic import BaseModel, Field, SecretStr


class AuthType(str, Enum):
    NONE = "none"
    API_KEY_HEADER = "api_key_header"
    API_KEY_QUERY = "api_key_query"
    BEARER = "bearer"
    BASIC = "basic"


class AuthConfig(BaseModel):
    """Authentication configuration for a REST API client."""

    auth_type: AuthType = Field(AuthType.NONE, description="Authentication method")
    api_key: Optional[SecretStr] = Field(None, description="API key or Bearer token")
    api_key_header: str = Field("X-Api-Key", description="Header name for API key auth")
    api_key_param: str = Field("apikey", description="Query param name for API key auth")
    username: Optional[str] = Field(None, description="Username for Basic auth")
    password: Optional[SecretStr] = Field(None, description="Password for Basic auth")

    class Config:
        frozen = True

    def build(self) -> Tuple[dict[str, str], dict[str, str], Any]:
        """
        Resolve auth config into (headers, query_params, httpx_auth).

        Returns
        -------
        Tuple[dict, dict, Any]
            headers, query_params, and httpx auth object (or None).
        """
        headers: dict[str, str] = {}
        params: dict[str, str] = {}
        auth: Any = None

        if self.auth_type == AuthType.BEARER:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        elif self.auth_type == AuthType.API_KEY_HEADER:
            headers[self.api_key_header] = self.api_key.get_secret_value()
        elif self.auth_type == AuthType.API_KEY_QUERY:
            params[self.api_key_param] = self.api_key.get_secret_value()
        elif self.auth_type == AuthType.BASIC:
            auth = (self.username, self.password.get_secret_value())

        return headers, params, auth
