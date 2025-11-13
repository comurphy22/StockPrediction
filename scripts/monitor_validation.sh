#!/bin/bash
# Monitor MVP Validation Progress

echo "========================================"
echo "MVP VALIDATION PROGRESS MONITOR"
echo "========================================"
echo ""

if [ ! -f "logs/mvp_validation.log" ]; then
    echo "❌ Log file not found. Script may not be running."
    exit 1
fi

# Get file size
SIZE=$(wc -c < logs/mvp_validation.log)
LINES=$(wc -l < logs/mvp_validation.log)

echo "Log file size: $SIZE bytes ($LINES lines)"
echo ""

# Check which stocks are complete
echo "📊 STOCKS COMPLETED:"
echo "--------------------"
grep -E "Summary for (NFLX|NVDA|BABA|QCOM|MU|TSLA|AAPL|MSFT|GOOGL|AMZN)" logs/mvp_validation.log | wc -l | xargs echo "Stocks completed:"
echo ""

# Check for any errors
ERROR_COUNT=$(grep -c "❌ Error" logs/mvp_validation.log || echo "0")
echo "Errors encountered: $ERROR_COUNT"
echo ""

# Show current status (last 20 lines)
echo "📝 RECENT OUTPUT:"
echo "--------------------"
tail -20 logs/mvp_validation.log
echo ""
echo "========================================"
echo "To view full log: cat logs/mvp_validation.log"
echo "To view live updates: tail -f logs/mvp_validation.log"
echo "========================================"
