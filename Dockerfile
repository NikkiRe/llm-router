FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ ./cmd/...

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/ /usr/local/bin/
COPY config.example.yaml /etc/llmrouter/config.yaml
EXPOSE 8080 9090
ENTRYPOINT ["/usr/local/bin/llmrouter", "-config", "/etc/llmrouter/config.yaml"]
