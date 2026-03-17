import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional

import httpx
from pydantic import BaseModel, Field

from gigaspatial.core.http.auth import AuthConfig, AuthType
from gigaspatial.core.http.pagination import BasePaginationStrategy

logger = logging.getLogger(__name__)


class RestApiClientConfig(BaseModel):
    """Top-level configuration for a REST API client."""

    base_url: str = Field(..., description="Base URL of the REST API")
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication config")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Max retries on transient errors")
    retry_backoff: float = Field(1.5, description="Exponential backoff multiplier")
    default_headers: dict[str, str] = Field(default_factory=dict)

    class Config:
        frozen = True


class BaseRestApiClient(ABC):
    """
    Abstract base class for REST API clients in GigaSpatial.

    Composes authentication (AuthConfig), retry logic, and pagination
    (BasePaginationStrategy) into a reusable context-managed client.
    Subclasses define the pagination strategy and any endpoint-specific methods.

    Parameters
    ----------
    config : RestApiClientConfig
        Pydantic configuration for the client.

    Examples
    --------
    >>> config = RestApiClientConfig(
    ...     base_url="https://api.example.com",
    ...     auth=AuthConfig(auth_type=AuthType.BEARER, api_key="secret"),
    ... )
    >>> with MyApiClient(config) as client:
    ...     for page in client.paginate("/schools", params={"country": "KE"}):
    ...         process(page)
    """

    def __init__(self, config: RestApiClientConfig) -> None:
        self.config = config
        self._client: Optional[httpx.Client] = None

    # ------------------------------------------------------------------
    # Pagination strategy — subclasses define this
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def pagination_strategy(self) -> BasePaginationStrategy:
        """Return the pagination strategy for this API."""
        ...

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _build_client(self) -> httpx.Client:
        auth_headers, auth_params, httpx_auth = self.config.auth.build()
        headers = {**self.config.default_headers, **auth_headers}
        return httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            params=auth_params,
            auth=httpx_auth,
            timeout=self.config.timeout,
        )

    def __enter__(self) -> "BaseRestApiClient":
        self._client = self._build_client()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Core request with retry + rate-limit handling
    # ------------------------------------------------------------------

    def request(self, method: str, endpoint: str, **kwargs: Any) -> httpx.Response:
        """
        Send an HTTP request with automatic retries and backoff.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.).
        endpoint : str
            API endpoint path relative to base_url.
        **kwargs
            Passed directly to httpx.Client.request().

        Returns
        -------
        httpx.Response

        Raises
        ------
        httpx.HTTPStatusError
            If the request fails after all retries.
        """
        assert self._client is not None, "Client not started — use as a context manager."
        delay = 1.0

        for attempt in range(1, self.config.max_retries + 1):
            response = self._client.request(method, endpoint, **kwargs)

            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", delay))
                logger.warning("Rate limited. Waiting %.1fs (attempt %d/%d)", wait, attempt, self.config.max_retries)
                time.sleep(wait)
                delay *= self.config.retry_backoff
                continue

            if response.status_code >= 500 and attempt < self.config.max_retries:
                logger.warning("Server error %d. Retrying in %.1fs (attempt %d/%d)", response.status_code, delay, attempt, self.config.max_retries)
                time.sleep(delay)
                delay *= self.config.retry_backoff
                continue

            response.raise_for_status()
            return response

        response.raise_for_status()
        return response

    def get(self, endpoint: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for GET requests."""
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for POST requests."""
        return self.request("POST", endpoint, **kwargs)

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def paginate(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Generator[list[dict], None, None]:
        """
        Yield pages of records using the configured pagination strategy.

        Parameters
        ----------
        endpoint : str
            API endpoint to paginate.
        params : dict, optional
            Initial query parameters.

        Yields
        ------
        list[dict]
            One page of records per iteration.
        """
        current_params = params or {}
        strategy = self.pagination_strategy

        while True:
            response = self.get(endpoint, params=current_params)
            records = strategy.extract_records(response)
            if not records:
                break
            yield records
            next_params = strategy.next_request(response, current_params)
            if next_params is None:
                break
            current_params = next_params
