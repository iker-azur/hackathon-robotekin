"""The localhost dashboard: a web app served on a thread beside the simulator.

``server.py`` builds the FastAPI app and runs it with uvicorn; ``static/index.html`` is the whole
front end, with no build step and no network access at run time.
"""