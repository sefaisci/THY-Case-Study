FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/frontend/streamlit

COPY frontend/streamlit/requirements.txt /app/frontend/streamlit/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/frontend/streamlit/requirements.txt

COPY frontend/streamlit /app/frontend/streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
