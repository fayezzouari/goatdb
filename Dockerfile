# syntax=docker/dockerfile:1

# ── build stage ───────────────────────────────────────────────────────────────
# Runs natively on the build host and cross-compiles for the target platform.
FROM --platform=$BUILDPLATFORM golang:1.23-alpine AS builder

ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev

WORKDIR /src

COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

COPY . .

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" \
      -o /out/goatdb ./cmd/goatdb

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM alpine:3.20

ARG VERSION=dev
ARG REVISION=unknown
ARG CREATED=unknown

LABEL org.opencontainers.image.title="goatdb" \
      org.opencontainers.image.description="A vector database built from scratch in Go" \
      org.opencontainers.image.source="https://github.com/fayezzouari/goatdb" \
      org.opencontainers.image.url="https://github.com/fayezzouari/goatdb" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}"

RUN apk add --no-cache ca-certificates && \
    addgroup -S goatdb && adduser -S goatdb -G goatdb && \
    mkdir -p /data && chown goatdb:goatdb /data

COPY --from=builder /out/goatdb /usr/local/bin/goatdb

USER goatdb

ENV GOATDB_ADDR=:8080 \
    GOATDB_DIR=/data

EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- "http://127.0.0.1:${GOATDB_ADDR##*:}/health" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/goatdb"]
