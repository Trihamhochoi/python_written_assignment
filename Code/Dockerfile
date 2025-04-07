# Use a lightweight Alpine image with SQLite
FROM alpine:latest

# Install sqlite3
RUN apk add --no-cache sqlite

# Set working directory
WORKDIR /db

# Default command
CMD [ "sqlite3" ]
