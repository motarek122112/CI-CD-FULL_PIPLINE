FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN echo "Preparing container..."
RUN echo "Downloading model for Run ID: ${RUN_ID}"

CMD ["sh", "-c", "echo Model container is ready for Run ID: ${RUN_ID}"]
