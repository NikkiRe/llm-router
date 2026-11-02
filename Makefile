MODEL ?= $(HOME)/models/model.gguf

.PHONY: build test lint bench-fake bench-llamacpp demo

build:
	go build -o bin/ ./cmd/...

test:
	go test -race ./...
	cd bench && uv run pytest -q

lint:
	test -z "$$(gofmt -l .)"
	go vet ./...
	go run honnef.co/go/tools/cmd/staticcheck@latest ./...
	cd bench && uv run ruff check . && uv run ruff format --check .

# Every policy against four simulated backends, two scenarios.
bench-fake: build
	cd bench && uv run python -m llmbench.run --no-build --backend fake --scenario docs --out results/fake-docs
	cd bench && uv run python -m llmbench.run --no-build --backend fake --scenario chat --out results/fake-chat \
		--rate 0.6 --doc-words 300

# Two llama.cpp servers with a real model: make bench-llamacpp MODEL=path/to/model.gguf
bench-llamacpp: build
	cd bench && uv run python -m llmbench.run --no-build --backend llamacpp --model $(MODEL) \
		--scenario docs --out results/llamacpp-docs \
		--instances 2 --slots 2 --docs 6 --doc-words 700 --requests 80 --rate 0.5 --max-tokens 16

demo:
	docker compose -f deploy/compose.yaml up --build
