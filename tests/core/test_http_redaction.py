"""Tests for URL and exception token redaction helpers."""

import pytest
import requests
import urllib3

from gigaspatial.core.http.redaction import redact_url, sanitize_exception


class TestRedactUrl:
    def test_masks_known_token_query_params_and_preserves_non_secret_params(self):
        url = (
            "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/0,0,5/512x512"
            "?access_token=pk.secret123&attribution=false&logo=false"
        )
        assert redact_url(url) == (
            "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/0,0,5/512x512"
            "?access_token=***&attribution=false&logo=false"
        )

    def test_masks_token_api_key_key_connectid_and_prefixed_variants(self):
        cases = [
            (
                "https://georepo.unicef.org/tiles?t=123&token=geo-secret&georepo_user_key=a@b.org",
                "https://georepo.unicef.org/tiles?t=123&token=***&georepo_user_key=a@b.org",
            ),
            (
                "https://pro.gegd.com/streaming/v1/ogc/wms?maxar_api_key=mx-secret&service=WMS",
                "https://pro.gegd.com/streaming/v1/ogc/wms?maxar_api_key=***&service=WMS",
            ),
            (
                "https://example.com/api?connectid=conn-999&layers=Maxar:Imagery",
                "https://example.com/api?connectid=***&layers=Maxar:Imagery",
            ),
            (
                "https://healthsites.io/api/v3/facilities/?api-key=hs-secret&page=1",
                "https://healthsites.io/api/v3/facilities/?api-key=***&page=1",
            ),
            (
                "https://example.com/v1?api_key=k1&key=k2&apikey=k3&quadkey=0123&format=json",
                "https://example.com/v1?api_key=***&key=***&apikey=***&quadkey=0123&format=json",
            ),
            (
                "https://example.com/v1?ACCESS_TOKEN=UPPER&ConnectId=MIXED&client_secret=shh",
                "https://example.com/v1?ACCESS_TOKEN=***&ConnectId=***&client_secret=***",
            ),
        ]
        for raw, expected in cases:
            assert redact_url(raw) == expected

    def test_masks_urls_embedded_in_error_text_and_extra_secrets(self):
        msg = (
            "401 Client Error: Unauthorized for url: "
            "https://api.mapbox.com/styles/v1/static?access_token=pk.abc&logo=false "
            "(raw token raw-secret-xyz)"
        )
        redacted = redact_url(msg, extra_secrets=("raw-secret-xyz",))
        assert "pk.abc" not in redacted
        assert "raw-secret-xyz" not in redacted
        assert "access_token=***&logo=false" in redacted
        assert "(raw token ***)" in redacted

    def test_leaves_plain_urls_without_secrets_unchanged(self):
        url = "https://example.com/data?country=KEN&page=2&format=geojson#section"
        assert redact_url(url) == url


class TestSanitizeException:
    def test_redacts_exception_args_response_url_and_cause_chain(self):
        secret = "top-secret-token-42"
        url = f"https://api.example.com/v1/item?access_token={secret}&format=json"

        resp = requests.Response()
        resp.status_code = 403
        resp.url = url
        req = requests.Request("GET", url).prepare()
        resp.request = req

        inner = urllib3.exceptions.MaxRetryError(
            None, f"/v1/item?access_token={secret}&format=json"
        )
        outer = requests.HTTPError(
            f"403 Client Error for url: {url}",
            request=req,
            response=resp,
        )
        outer.__cause__ = inner

        sanitize_exception(outer, extra_secrets=(secret,))

        assert secret not in str(outer)
        assert secret not in str(outer.__cause__)
        assert secret not in outer.request.url
        assert secret not in outer.response.url
        assert "access_token=***&format=json" in str(outer)


class TestRestApiClientAndHealthSitesRedaction:
    def test_base_rest_api_client_redacts_query_api_key_on_http_error(self):
        import httpx
        import pytest

        from gigaspatial.core.http import (
            AuthConfig,
            AuthType,
            BaseRestApiClient,
            PageNumberPagination,
            RestApiClientConfig,
        )

        secret = "hs-query-secret-777"

        class _TestClient(BaseRestApiClient):
            @property
            def pagination_strategy(self):
                return PageNumberPagination()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, request=request)

        cfg = RestApiClientConfig(
            base_url="https://healthsites.io/api/v3/facilities",
            auth=AuthConfig(
                auth_type=AuthType.API_KEY_QUERY,
                api_key=secret,
                api_key_param="api-key",
            ),
            max_retries=1,
        )
        client = _TestClient(cfg)
        auth_headers, auth_params, httpx_auth = cfg.auth.build()
        client._client = httpx.Client(
            base_url=cfg.base_url,
            headers=auth_headers,
            params=auth_params,
            auth=httpx_auth,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                client.get("/statistic/", params={"country": "Kenya"})
        finally:
            client._client.close()

        assert secret not in str(exc_info.value)
        assert secret not in str(exc_info.value.request.url)
        assert "api-key=***" in str(exc_info.value)
        assert "country=Kenya" in str(exc_info.value)

    def test_healthsites_fetcher_repr_masks_api_key(self):
        from gigaspatial.fetchers.healthsites import HealthSitesFetcher

        secret = "hs-query-secret-777"
        fetcher = HealthSitesFetcher(api_key=secret)
        assert secret not in repr(fetcher)

    @staticmethod
    def _build_header_auth_fetcher(kind: str, secret: str):
        if kind == "peeringdb":
            from gigaspatial.fetchers.peeringdb import PeeringDBFetcher

            return PeeringDBFetcher(api_key=secret)
        if kind == "giga_countries":
            from gigaspatial.fetchers.giga.countries import GigaCountriesFetcher

            return GigaCountriesFetcher(api_key=secret)
        if kind == "giga_school_locations":
            from gigaspatial.fetchers.giga.school_locations import (
                GigaSchoolLocationFetcher,
            )

            return GigaSchoolLocationFetcher(country="KEN", api_key=secret)
        if kind == "giga_school_measurements":
            from gigaspatial.fetchers.giga.school_measurements import (
                GigaSchoolMeasurementsFetcher,
            )

            return GigaSchoolMeasurementsFetcher(
                country="KEN",
                start_date="2024-01-01",
                end_date="2024-01-31",
                api_key=secret,
            )
        if kind == "giga_school_profile":
            from gigaspatial.fetchers.giga.school_profile import (
                GigaSchoolProfileFetcher,
            )

            return GigaSchoolProfileFetcher(country="KEN", api_key=secret)
        raise ValueError(f"Unknown fetcher kind: {kind}")

    @pytest.mark.parametrize(
        "kind",
        [
            "peeringdb",
            "giga_countries",
            "giga_school_locations",
            "giga_school_measurements",
            "giga_school_profile",
        ],
    )
    def test_header_auth_fetchers_repr_omit_api_key(self, kind):
        secret = "header-auth-secret-888"
        fetcher = self._build_header_auth_fetcher(kind, secret)
        assert secret not in repr(fetcher)
