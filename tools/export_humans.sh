#!/usr/bin/env bash
# Exports human answers from the game's D1 database into data/human_bridges.json.
# Testers (players.tester = 1) and skipped pairs are left out.
# Run from the repo root, logged in with `npx wrangler login`.
# The file contains free-text answers: keep it out of git (data/ is ignored).
set -euo pipefail
mkdir -p data
npx wrangler d1 execute bridges --remote --json --command "
SELECT t.code, t.pair, t.set_name AS \"set\", t.level, t.dir, t.lang, t.from_text AS \"from\", t.to_text AS \"to\",
       t.final, t.stones, t.difficulty, t.ai_requested, t.voice, t.pasted, t.submit_ms, t.position, t.at,
       w.words AS warmup_words, e.data AS end_answers
FROM trials t
JOIN players p ON p.code = t.code
LEFT JOIN warmups w ON w.code = t.code
LEFT JOIN end_answers e ON e.code = t.code
WHERE p.tester = 0 AND t.skipped = 0 AND t.final IS NOT NULL AND t.final <> ''
ORDER BY t.id" > data/human_bridges.raw.json
python3 -c "
import json; d = json.load(open('data/human_bridges.raw.json'))[0]['results']
json.dump(d, open('data/human_bridges.json', 'w'), ensure_ascii=False, indent=1)
print(len(d), 'answers saved to data/human_bridges.json')"
rm data/human_bridges.raw.json
