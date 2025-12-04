FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY server/ server/
COPY dashboard/ dashboard/
COPY data/ data/
COPY knowledge_base/ knowledge_base/
COPY scripts/ scripts/
COPY entrypoint.sh .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
