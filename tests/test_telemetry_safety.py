from forge.telemetry import _sanitize_value


def test_sanitize_value_redacts_secret_keys_recursively():
    value = {
        "github_token": "ghp_private",
        "api_key": "api-private",
        "nested": {
            "password": "private",
            "accessKey": "access-private",
            "session_id": "session-private",
            "phase": "load_model",
        },
    }

    assert _sanitize_value(value) == {
        "github_token": "<redacted>",
        "api_key": "<redacted>",
        "nested": {
            "password": "<redacted>",
            "accessKey": "<redacted>",
            "session_id": "<redacted>",
            "phase": "load_model",
        },
    }


def test_sanitize_value_strips_signed_url_query_but_keeps_path():
    message = (
        "failed https://storage.example/model.bin?X-Amz-Signature=secret&token=x; "
        "retry https://huggingface.co/org/repo."
    )

    sanitized = _sanitize_value(message, key="error")

    assert "secret" not in sanitized
    assert "token=x" not in sanitized
    assert "https://storage.example/model.bin" in sanitized
    assert "https://huggingface.co/org/repo." in sanitized


def test_sanitize_value_redacts_bearer_tokens():
    assert _sanitize_value("Authorization: Bearer abc.def-123", key="error") == (
        "Authorization: <redacted>"
    )


def test_sanitize_value_redacts_inline_assignments_and_non_http_signed_urls():
    message = (
        "api_key=super-secret cookie:chocolate "
        "s3://private-bucket/model.bin?X-Amz-Credential=private&X-Amz-Signature=x"
    )

    sanitized = _sanitize_value(message, key="error")

    assert "super-secret" not in sanitized
    assert "chocolate" not in sanitized
    assert "private&" not in sanitized
    assert "s3://private-bucket/model.bin" in sanitized


def test_sanitize_value_strips_url_userinfo():
    sanitized = _sanitize_value(
        "failed https://user:password@example.com:8443/private/file?token=x",
        key="error",
    )

    assert sanitized == "failed https://example.com:8443/private/file"


def test_sanitize_value_redacts_auth_cookie_and_common_token_forms():
    message = (
        "Authorization: Basic dXNlcjpwYXNz\n"
        "Cookie: session=one; csrf=two; preference=three\n"
        "client_secret=client-value refresh_token=refresh-value "
        "access_token=access-value AWS_SECRET_ACCESS_KEY=aws-value "
        "github_pat_1234567890abcdefghijklmnop "
        "hf_1234567890abcdefghijklmnop"
    )

    sanitized = _sanitize_value(message, key="tail")

    for secret in (
        "dXNlcjpwYXNz",
        "session=one",
        "csrf=two",
        "client-value",
        "refresh-value",
        "access-value",
        "aws-value",
        "github_pat_",
        "hf_1234567890",
    ):
        assert secret not in sanitized


def test_sanitize_value_does_not_redact_token_count_metadata():
    assert _sanitize_value(61556, key="tokenized") == 61556
    assert _sanitize_value(32768, key="tokens_per_step_cap") == 32768
