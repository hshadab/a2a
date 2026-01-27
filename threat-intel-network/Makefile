.PHONY: setup setup-zkml build run stop logs clean dev test validate

# Setup the project
setup:
	@echo "Setting up Threat Intelligence Network..."
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installing UI dependencies..."
	cd ui && npm install
	@echo ""
	@echo "Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make setup-zkml' to install the zkML prover"
	@echo "  3. Run 'make dev' to start development environment"

# Setup zkML prover binary
setup-zkml:
	@echo "Installing zkML (Jolt Atlas) prover binary..."
	./scripts/setup_zkml.sh

# Validate production configuration
validate:
	@echo "Validating configuration..."
	@python3 -c "from shared.config import config; errors = config.validate_production_requirements(); print('\\n'.join(errors) if errors else 'All validations passed!')"

# Build all containers
build:
	docker-compose build

# Run the full stack
run:
	docker-compose up -d
	@echo "Services starting..."
	@echo "  Scout Agent: http://localhost:8000"
	@echo "  Policy Agent: http://localhost:8001"
	@echo "  Analyst Agent: http://localhost:8002"
	@echo "  UI Dashboard: http://localhost:3001"

# Stop all containers
stop:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

logs-scout:
	docker-compose logs -f scout

logs-policy:
	docker-compose logs -f policy

logs-analyst:
	docker-compose logs -f analyst

# Clean up
clean:
	docker-compose down -v
	rm -rf ui/.next ui/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} +

# Development mode (run services locally)
dev:
	@echo "Starting development environment..."
	@echo "Make sure PostgreSQL and Redis are running locally"
	@trap 'kill 0' EXIT; \
	cd agents/policy && python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload & \
	cd agents/analyst && python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload & \
	cd agents/scout && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload & \
	cd ui && npm run dev & \
	wait

# Run just the UI in dev mode
dev-ui:
	cd ui && npm run dev

# Trigger a batch manually
trigger:
	curl -X POST http://localhost:8000/trigger

# Check health
health:
	@echo "Scout:" && curl -s http://localhost:8000/health | jq
	@echo "Policy:" && curl -s http://localhost:8001/health | jq
	@echo "Analyst:" && curl -s http://localhost:8002/health | jq

# Get stats
stats:
	curl -s http://localhost:8000/stats | jq

# Initialize database (run migrations)
db-init:
	docker-compose exec db psql -U threat_intel -d threat_intel -c "SELECT 1"
	@echo "Database ready"

# Test the x402 payment flow
test-payment:
	@echo "Testing x402 payment flow..."
	python scripts/test_x402.py
