FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Set environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=300
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application code and config
COPY config/ config/
COPY server/ server/
COPY data/ data/
COPY knowledge_base/ knowledge_base/
COPY scripts/ scripts/
COPY entrypoint.sh .

# Expose ports (HTTP and SMTP)
EXPOSE 8000 8025

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
