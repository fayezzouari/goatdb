# Benchmark client: runs inside the compose network next to the engine under test,
# so Docker Desktop's host port forwarding is not in the query path.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /bench
CMD ["python", "run.py", "--help"]
