"""Helpers for redacting API tokens and secrets from URLs, logs, and exceptions."""

from __future__ import annotations

import re
from typing import Iterable, Optional

_TOKEN_PARAM_RE = re.compile(
    r"([?&](?:(?:[a-z0-9_-]+[_-])?(?:token|api[_-]?key|secret)|key|connect[_-]?id)=)[^&#\s'\"<>)\]}]+",
    re.IGNORECASE,
)


def redact_url(
    text: str,
    extra_secrets: Iterable[Optional[str]] = (),
) -> str:
    """Mask token-like query parameters and optional literal secrets in a string.

    Non-secret query parameters are left untouched.
    """
    redacted = _TOKEN_PARAM_RE.sub(r"\1***", text)
    for secret in extra_secrets:
        if secret:
            redacted = redacted.replace(secret, "***")
    return redacted


def _redact_url_attr(obj: object, secrets: tuple[str, ...]) -> None:
    """Redact a ``url`` attribute on a request, response, or exception object."""
    url = getattr(obj, "url", None)
    if isinstance(url, str):
        obj.url = redact_url(url, secrets)
    elif url is not None:
        url_str = str(url)
        redacted = redact_url(url_str, secrets)
        if redacted != url_str:
            try:
                obj.url = type(url)(redacted)
            except Exception:
                pass


def sanitize_exception(
    exc: BaseException,
    extra_secrets: Iterable[Optional[str]] = (),
) -> BaseException:
    """Redact tokens from an exception's message, cause/context chain, and URLs."""
    secrets = tuple(s for s in extra_secrets if s)
    seen: set[int] = set()

    def _visit(err: Optional[BaseException]) -> None:
        if err is None or id(err) in seen:
            return
        seen.add(id(err))

        _visit(err.__cause__)
        _visit(err.__context__)

        _redact_url_attr(err, secrets)

        request = getattr(err, "request", None)
        if request is not None:
            _redact_url_attr(request, secrets)

        response = getattr(err, "response", None)
        if response is not None:
            _redact_url_attr(response, secrets)

        if isinstance(err, OSError) and isinstance(err.strerror, str):
            err.strerror = redact_url(err.strerror, secrets)

        if not isinstance(err, UnicodeError) and getattr(err, "args", None):
            new_args = []
            for arg in err.args:
                if isinstance(arg, str):
                    new_args.append(redact_url(arg, secrets))
                elif isinstance(arg, BaseException):
                    _visit(arg)
                    arg_str = str(arg)
                    redacted = redact_url(arg_str, secrets)
                    new_args.append(redacted if redacted != arg_str else arg)
                else:
                    new_args.append(arg)
            err.args = tuple(new_args)

    _visit(exc)
    return exc
