import requests


class EarthdataSession(requests.Session):
    """
    Custom requests.Session for NASA Earthdata authentication.

    Maintains Authorization headers through redirects to/from Earthdata hosts.
    This is required because Earthdata uses multiple redirect domains during authentication.
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        """Keep auth header on redirects to/from Earthdata host."""
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            # remove Authorization only if redirecting *away from* Earthdata
            if (
                (original_parsed.hostname != redirect_parsed.hostname)
                and redirect_parsed.hostname != self.AUTH_HOST
                and original_parsed.hostname != self.AUTH_HOST
            ):
                del headers["Authorization"]
