#!/bin/bash
# Rebuild everything: data -> HTML -> PNG exports -> index (thumbnails) -> checks. Needs /workspace/pres-build/node_modules.
set -e
B=$(cd "$(dirname "$0")" && pwd); P=$(dirname "$B"); N=/workspace/pres-build
cd "$B"
python3 prep_graph.py >/dev/null; python3 prep_results.py >/dev/null
python3 assemble.py graph results
rm -f "$P"/png/*.png
cd $N
node graphshots.js "$P/graph3d.html" "$P/png"
node shots.js "$P/results.html" "$P/png" "div.plot" "chart_" 2
node shots.js "$P/results.html" "$P/png" "section[id]" "results_section_" 2
cd "$B"; python3 assemble.py index; python3 rename_png.py
cd $N; node check.js "$P/index.html" "$P/png/00_index_landing.png" 1600 1000
