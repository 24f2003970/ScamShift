FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SCAMSHIFT_STATE=/app/runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 scamshift \
    && useradd --uid 10001 --gid scamshift --no-create-home scamshift \
    && install -d -o scamshift -g scamshift /app/runtime
COPY main.py train.py ./
COPY scamshift/ scamshift/
COPY web/ web/
COPY data/ data/
USER 10001:10001
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
