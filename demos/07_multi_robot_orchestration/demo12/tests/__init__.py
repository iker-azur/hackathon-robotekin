"""Test support for demo12.

A package, and not just a directory, because ``FakeCell`` -- the scripted physics that lets
the orchestrator and the HTTP surface both be tested without a GPU -- is now used by more
than one test module. Without this, ``test_server`` can only reach it through a ``sys.path``
trick that depends on pytest's import mode.
"""