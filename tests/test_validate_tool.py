import logging

import pytest

from optulus_anchor import validate_tool


class GetCustomerParams:
    pass


class CustomerResponse:
    pass


@validate_tool(params_schema=GetCustomerParams, response_schema=CustomerResponse)
def some_function(x: int) -> int:
    return x * 2


def test_validate_tool_logs_schemas_on_call(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="optulus_anchor.decorators"):
        assert some_function(3) == 6

    messages = " ".join(r.message for r in caplog.records)
    assert "some_function" in messages
    assert "GetCustomerParams" in messages
    assert "CustomerResponse" in messages
