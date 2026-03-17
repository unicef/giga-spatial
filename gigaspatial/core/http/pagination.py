from abc import ABC, abstractmethod
from typing import Any, Generator, Optional

import httpx


class BasePaginationStrategy(ABC):
    """
    Abstract base for pagination strategies.

    Subclass this to implement cursor-based, offset/limit,
    page-number, or Link-header pagination.
    """

    @abstractmethod
    def next_request(
        self,
        response: httpx.Response,
        current_params: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Return updated params for the next page, or None if exhausted.

        Parameters
        ----------
        response : httpx.Response
            The most recent response from the API.
        current_params : dict
            The params used for the current request.

        Returns
        -------
        Optional[dict]
            Params for the next request, or None to stop pagination.
        """
        ...

    @abstractmethod
    def extract_records(self, response: httpx.Response) -> list[dict]:
        """Extract the list of records from a response."""
        ...


class OffsetPagination(BasePaginationStrategy):
    """
    Standard offset/limit pagination.

    Parameters
    ----------
    page_size : int
        Number of records per page.
    offset_param : str
        Query parameter name for the offset.
    limit_param : str
        Query parameter name for the limit/page size.
    records_key : str
        JSON key containing the list of records.
    """

    def __init__(
        self,
        page_size: int = 100,
        offset_param: str = "offset",
        limit_param: str = "limit",
        records_key: str = "results",
    ) -> None:
        self.page_size = page_size
        self.offset_param = offset_param
        self.limit_param = limit_param
        self.records_key = records_key

    def next_request(
        self,
        response: httpx.Response,
        current_params: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        records = self.extract_records(response)
        if len(records) < self.page_size:
            return None  # last page
        next_offset = current_params.get(self.offset_param, 0) + self.page_size
        return {**current_params, self.offset_param: next_offset, self.limit_param: self.page_size}

    def extract_records(self, response: httpx.Response) -> list[dict]:
        return response.json().get(self.records_key, [])


class CursorPagination(BasePaginationStrategy):
    """
    Cursor-based pagination (e.g. GitHub, Mapbox APIs).

    Parameters
    ----------
    cursor_response_key : str
        JSON key in the response body containing the next cursor.
    cursor_param : str
        Query parameter name to pass the cursor on the next request.
    records_key : str
        JSON key containing the list of records.
    """

    def __init__(
        self,
        cursor_response_key: str = "next_cursor",
        cursor_param: str = "cursor",
        records_key: str = "results",
    ) -> None:
        self.cursor_response_key = cursor_response_key
        self.cursor_param = cursor_param
        self.records_key = records_key

    def next_request(
        self,
        response: httpx.Response,
        current_params: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        cursor = response.json().get(self.cursor_response_key)
        if not cursor:
            return None
        return {**current_params, self.cursor_param: cursor}

    def extract_records(self, response: httpx.Response) -> list[dict]:
        return response.json().get(self.records_key, [])

# gigaspatial/core/http/pagination.py  (add this class)

class PageNumberPagination(BasePaginationStrategy):
    """
    Page-number pagination (page=1, page=2, ...).

    Parameters
    ----------
    page_size : int
        Number of records per page.
    page_param : str
        Query parameter name for the page number.
    size_param : str
        Query parameter name for the page size.
    records_key : str
        JSON key containing the list of records.
    """

    def __init__(
        self,
        page_size: int = 1000,
        page_param: str = "page",
        size_param: str = "size",
        records_key: str = "data",
    ) -> None:
        self.page_size = page_size
        self.page_param = page_param
        self.size_param = size_param
        self.records_key = records_key

    def next_request(
        self,
        response: httpx.Response,
        current_params: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        records = self.extract_records(response)
        if len(records) < self.page_size:
            return None  # partial page — last page reached
        next_page = current_params.get(self.page_param, 1) + 1
        return {**current_params, self.page_param: next_page}

    def extract_records(self, response: httpx.Response) -> list[dict]:
        return response.json().get(self.records_key, [])
