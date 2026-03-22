# ── build stage ───────────────────────────────────────────────────────────────
FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o goatdb ./cmd/goatdb

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM alpine:3.19

RUN apk add --no-cache ca-certificates && \
    addgroup -S goatdb && adduser -S goatdb -G goatdb

WORKDIR /app

COPY --from=builder /app/goatdb .

RUN mkdir -p /data && chown goatdb:goatdb /data

USER goatdb

EXPOSE 8080

VOLUME ["/data"]

ENTRYPOINT ["./goatdb"]
CMD ["-addr", ":8080", "-dir", "/data"]
