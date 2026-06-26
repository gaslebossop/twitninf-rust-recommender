#!/bin/bash
# 🚀 Deploy Script for TwitNinf Rust Recommender on VPS

set -e

echo "📦 Building Rust Recommender..."
cargo build --release

echo "✅ Build complete!"
echo ""
echo "🔧 Next steps on your VPS (debian@your_vps):"
echo ""
echo "1. Copy to VPS:"
echo "   scp -r ./target/release/twitninf-recommender debian@your_vps:/home/debian/rust-recommender/"
echo "   scp ./twitninf-rust-recommender.service debian@your_vps:/tmp/"
echo ""
echo "2. On VPS, install the systemd service:"
echo "   ssh debian@your_vps"
echo "   sudo cp /tmp/twitninf-rust-recommender.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable twitninf-rust-recommender"
echo "   sudo systemctl start twitninf-rust-recommender"
echo ""
echo "3. Check logs in real-time:"
echo "   sudo journalctl -u twitninf-rust-recommender -f"
echo ""
echo "📊 Done! Your recommender is running with debug logs enabled."
