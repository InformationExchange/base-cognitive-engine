#!/bin/bash
# =============================================================================
# BAIS Cognitive Governance Engine - Deployment Script
# Phase 29: Production Deployment
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE="bais-governance"
DOCKER_TAG="${BAIS_VERSION:-29.0.0}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Pre-deployment Checks
# =============================================================================
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose installed"
    
    # Check environment file
    if [ ! -f "$PROJECT_DIR/.env" ] && [ ! -f "$PROJECT_DIR/env.example" ]; then
        log_warning "No .env file found. Using defaults."
    fi
    
    log_success "Prerequisites check passed"
}

# =============================================================================
# Build
# =============================================================================
build() {
    log_info "Building Docker image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    
    cd "$PROJECT_DIR"
    
    docker build \
        -t "${DOCKER_IMAGE}:${DOCKER_TAG}" \
        -t "${DOCKER_IMAGE}:latest" \
        -f Dockerfile \
        .
    
    log_success "Docker image built successfully"
}

# =============================================================================
# Deploy
# =============================================================================
deploy_local() {
    log_info "Deploying locally with Docker Compose..."
    
    cd "$PROJECT_DIR"
    
    # Stop existing containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Start services
    docker-compose up -d
    
    log_success "Services started"
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    check_health
}

deploy_production() {
    log_info "Deploying to production..."
    
    cd "$PROJECT_DIR"
    
    # Use production profile
    docker-compose --profile production up -d
    
    log_success "Production deployment complete"
}

# =============================================================================
# Health Check
# =============================================================================
check_health() {
    log_info "Checking service health..."
    
    local max_attempts=30
    local attempt=1
    local health_url="http://localhost:${BAIS_PORT:-8000}/health"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            log_success "Service is healthy!"
            curl -s "$health_url" | python3 -m json.tool 2>/dev/null || curl -s "$health_url"
            return 0
        fi
        
        log_info "Waiting for service... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_error "Service failed to become healthy"
    docker-compose logs --tail=50
    return 1
}

# =============================================================================
# Verify Deployment
# =============================================================================
verify_deployment() {
    log_info "Running deployment verification..."
    
    cd "$PROJECT_DIR/src"
    
    python3 -c "
import asyncio
from core.deployment import DeploymentVerifier

async def verify():
    verifier = DeploymentVerifier('http://localhost:${BAIS_PORT:-8000}')
    results = await verifier.verify_all()
    print(f\"Passed: {results['passed']}/{results['total']}\")
    print(f\"Success Rate: {results['success_rate']:.1f}%\")
    for r in results['results']:
        status = '✓' if r['status'] == 'pass' else '✗'
        print(f\"  {status} {r['test']}: {r.get('status_code', r.get('error', 'N/A'))}\")
    return results['success_rate'] >= 80

if asyncio.run(verify()):
    exit(0)
else:
    exit(1)
" || log_warning "Verification tests not all passing"
    
    log_success "Deployment verification complete"
}

# =============================================================================
# Logs
# =============================================================================
show_logs() {
    cd "$PROJECT_DIR"
    docker-compose logs -f
}

# =============================================================================
# Stop
# =============================================================================
stop() {
    log_info "Stopping services..."
    cd "$PROJECT_DIR"
    docker-compose down
    log_success "Services stopped"
}

# =============================================================================
# Status
# =============================================================================
status() {
    cd "$PROJECT_DIR"
    docker-compose ps
}

# =============================================================================
# Main
# =============================================================================
print_usage() {
    echo "BAIS Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  check       Check prerequisites"
    echo "  build       Build Docker image"
    echo "  deploy      Deploy locally (default)"
    echo "  production  Deploy to production"
    echo "  health      Check service health"
    echo "  verify      Run deployment verification"
    echo "  logs        Show service logs"
    echo "  stop        Stop all services"
    echo "  status      Show service status"
    echo ""
}

case "${1:-deploy}" in
    check)
        check_prerequisites
        ;;
    build)
        check_prerequisites
        build
        ;;
    deploy)
        check_prerequisites
        build
        deploy_local
        verify_deployment
        ;;
    production)
        check_prerequisites
        build
        deploy_production
        verify_deployment
        ;;
    health)
        check_health
        ;;
    verify)
        verify_deployment
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

