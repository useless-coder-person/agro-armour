#!/usr/bin/env bash
# KB sources -> $AGROARMOUR_SOURCES/<Crop>.txt, used by kb_lint.py to check doses verbatim.
set -euo pipefail
S=${AGROARMOUR_SOURCES:-$HOME/cropdoc/sources}; PY=${PY:-$HOME/cropdoc/venv/bin/python}
mkdir -p "$S"
for c in Rice Maize Tomato Potato Wheat Cotton Groundnut Mango; do
  [ -s "$S/$c.pdf" ] || curl -sSL --max-time 300 -o "$S/$c.pdf" "https://www.niphm.gov.in/IPMPackages/$c.pdf"
done
curl -sSL --max-time 120 -o "$S/banned_31.10.2026.pdf" \
  "https://ppqs.gov.in/sites/default/files/list_of_pesticides_which_are_banned_refused_registration_and_restricted_in_use%20%2831.10.26%29.pdf"
for f in "$S"/*.pdf; do
  "$PY" -c "import pypdf,sys; r=pypdf.PdfReader(sys.argv[1]); print('\n'.join(f'=== page {i+1}\n'+(p.extract_text() or '') for i,p in enumerate(r.pages)))" "$f" \
    | tr -d '\000' | tr -cd '\11\12\15\40-\176' > "${f%.pdf}.txt"
done
ls -la "$S"/*.txt
