.PHONY: docker-build docker-run docker-benchmark help

# Default target when just running `make`
help:
	@echo "BeyondML Makefile commands:"
	@echo ""
	@echo "  make docker-build       - Build the BeyondML Docker image"
	@echo "  make docker-run         - Run the interactive Textual TUI via Docker"
	@echo "  make docker-benchmark   - Run the PMLB benchmarking suite via Docker"

docker-build:
	docker-compose build

docker-run:
	docker-compose run --rm beyondml beyondml run

docker-benchmark:
	docker-compose run --rm beyondml beyondml benchmark
