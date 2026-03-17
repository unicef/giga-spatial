# ---------------------------------------------------------------------------
# Internal API client — not part of the public interface
# ---------------------------------------------------------------------------

from gigaspatial.core.http import BaseRestApiClient, RestApiClientConfig, PageNumberPagination

class GigaApiClient(BaseRestApiClient):
    """
    Internal HTTP client for the Giga API.

    Uses Bearer token auth and page-number pagination.
    """

    def __init__(self, config: RestApiClientConfig, page_size: int) -> None:
        super().__init__(config)
        self._page_size = page_size

    @property
    def pagination_strategy(self) -> PageNumberPagination:
        return PageNumberPagination(
            page_size=self._page_size,
            page_param="page",
            size_param="size",
            records_key="data",
        )
