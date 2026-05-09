FROM python:3.11-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/*.py /app/
COPY app/admin /app/admin
COPY app/agent /app/agent
COPY app/auth /app/auth
COPY app/billing /app/billing
COPY app/cases /app/cases
COPY app/db /app/db
COPY app/preferences /app/preferences
COPY app/rag /app/rag
COPY scripts /app/scripts
COPY data /app/data
COPY tests /app/tests
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
