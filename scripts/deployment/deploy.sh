#!/bin/bash
# Enhanced Nanpin Bot Deployment Script

set -e

echo "🚀 Enhanced Nanpin Bot Deployment Starting..."

# Configuration
PROJECT_DIR="/home/tetsu/Documents/nanpin_bot"
BACKUP_DIR="$PROJECT_DIR/data/backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_DIR/data/logs/system/deployment.log"

# Create directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🔧 Starting deployment process..."

# Health check function
health_check() {
    log "🔍 Running health checks..."
    
    # Check Python dependencies
    if ! python3 -c "import asyncio, aiohttp, websockets, yaml" 2>/dev/null; then
        log "❌ Python dependencies missing"
        return 1
    fi
    
    # Check configuration files
    if [[ ! -f "$PROJECT_DIR/config/enhanced_nanpin_config.yaml" ]]; then
        log "❌ Main configuration file missing"
        return 1
    fi
    
    # Check API credentials
    if [[ -z "$BACKPACK_API_KEY" || -z "$BACKPACK_SECRET_KEY" ]]; then
        log "❌ API credentials not set"
        return 1
    fi
    
    log "✅ Health checks passed"
    return 0
}

# Backup function
backup_current() {
    log "💾 Creating backup..."
    
    # Backup critical files
    cp -r "$PROJECT_DIR/src" "$BACKUP_DIR/"
    cp -r "$PROJECT_DIR/config" "$BACKUP_DIR/"
    cp "$PROJECT_DIR/.env" "$BACKUP_DIR/" 2>/dev/null || log "⚠️ No .env file to backup"
    
    log "✅ Backup created at $BACKUP_DIR"
}

# Update dependencies
update_dependencies() {
    log "📦 Updating dependencies..."
    cd "$PROJECT_DIR"
    
    pip3 install -r requirements.txt --upgrade
    
    log "✅ Dependencies updated"
}

# Start service
start_service() {
    log "🔄 Starting Enhanced Nanpin Bot..."
    
    cd "$PROJECT_DIR"
    
    # Kill existing process if running
    pkill -f "launch_enhanced_nanpin_bot.py" || true
    sleep 2
    
    # Start new process
    nohup python3 launch_enhanced_nanpin_bot.py > "$PROJECT_DIR/data/logs/system/bot.log" 2>&1 &
    
    BOT_PID=$!
    log "✅ Bot started with PID: $BOT_PID"
    
    # Wait a moment and check if still running
    sleep 5
    if kill -0 "$BOT_PID" 2>/dev/null; then
        log "✅ Bot is running successfully"
        echo "$BOT_PID" > "$PROJECT_DIR/bot.pid"
    else
        log "❌ Bot failed to start properly"
        return 1
    fi
}

# Verify deployment
verify_deployment() {
    log "🧪 Verifying deployment..."
    
    # Check if bot process is running
    if [[ -f "$PROJECT_DIR/bot.pid" ]]; then
        BOT_PID=$(cat "$PROJECT_DIR/bot.pid")
        if kill -0 "$BOT_PID" 2>/dev/null; then
            log "✅ Bot process is running (PID: $BOT_PID)"
        else
            log "❌ Bot process is not running"
            return 1
        fi
    else
        log "❌ No PID file found"
        return 1
    fi
    
    # Check log files for errors
    if tail -n 10 "$PROJECT_DIR/data/logs/system/bot.log" | grep -q "ERROR\|CRITICAL"; then
        log "⚠️ Errors detected in bot logs"
        tail -n 20 "$PROJECT_DIR/data/logs/system/bot.log"
    else
        log "✅ No critical errors in logs"
    fi
    
    log "✅ Deployment verification completed"
}

# Main deployment flow
main() {
    log "🎯 Enhanced Nanpin Bot Deployment v1.3.0"
    
    # Pre-deployment checks
    health_check || exit 1
    
    # Create backup
    backup_current
    
    # Update system
    update_dependencies
    
    # Deploy
    start_service || exit 1
    
    # Verify
    verify_deployment || exit 1
    
    log "🎉 Deployment completed successfully!"
    log "📊 Bot Status: RUNNING"
    log "📁 Logs: $PROJECT_DIR/data/logs/system/bot.log"
    log "🔧 PID: $(cat $PROJECT_DIR/bot.pid 2>/dev/null || echo 'Unknown')"
}

# Run main function
main "$@"