"""Make pytest honour this suite's own ``check()`` helper.

Both test modules record an assertion by appending to a module-level ``failures`` list and only
their ``main()`` reports it. Under pytest that list was never read, so a test whose only
assertions were ``check(...)`` calls passed whatever it found. This fixture fails the test that
added to the list.
"""
import pytest


@pytest.fixture(autouse=True)
def fail_on_recorded_check_failures(request):
    recorded = getattr(request.module, "failures", None)
    before = len(recorded) if isinstance(recorded, list) else 0
    yield
    if isinstance(recorded, list) and len(recorded) > before:
        pytest.fail("check() failed: " + "; ".join(recorded[before:]), pytrace=False)
