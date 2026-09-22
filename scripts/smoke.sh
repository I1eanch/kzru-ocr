#!/usr/bin/env bash
# Дымовая проверка собранного production-образа.
#
# Проверяет ровно те пути, которые менялись при устранении замечаний ревью:
# разделение liveness и readiness, лимит загрузки, ограниченная очередь
# заданий, коды ошибок, структура полей, отсутствие dev-зависимостей в
# production-образе и работа не от root.
#
#   scripts/smoke.sh [образ]
#
# Возвращает ненулевой код при первом же расхождении.
set -euo pipefail

IMAGE="${1:-kzru-ocr:latest}"
PORT="${SMOKE_PORT:-18099}"
NAME="kzru-smoke-$$"
# Отслеживаемая фикстура, а не игнорируемый `bench/scans/`: проверка обязана
# запускаться в свежем клоне без локальных артефактов.
PDF="${SMOKE_PDF:-tests/fixtures/one_page.pdf}"

pass() { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== образ $IMAGE"

echo "-- состав образа"
docker run --rm "$IMAGE" python -c "
import importlib.util, sys
forbidden = [m for m in ('pymupdf', 'jiwer', 'pytest') if importlib.util.find_spec(m)]
sys.exit('в production-образе есть dev-зависимости: %s' % forbidden if forbidden else 0)
" && pass "нет dev-зависимостей" || fail "dev-зависимости в production-образе"

docker run --rm "$IMAGE" sh -c '[ ! -d /app/bench ]' \
  && pass "каталог bench не поставляется" || fail "bench попал в production-образ"

docker run --rm "$IMAGE" id | grep -qv 'uid=0(root)' \
  && pass "процесс не от root: $(docker run --rm "$IMAGE" id -un)" || fail "контейнер работает от root"

docker run --rm "$IMAGE" python -c "import service.app; import ocr.cli" \
  && pass "сервис и CLI импортируются" || fail "импорт не прошёл"

echo "-- запуск сервиса"
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8099" --memory=2g "$IMAGE" >/dev/null
BASE="http://127.0.0.1:$PORT"

for _ in $(seq 1 60); do
  if curl -fsS "$BASE/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

[ "$(code "$BASE/healthz")" = "200" ] && pass "liveness отвечает 200" || fail "liveness не отвечает"
[ "$(code "$BASE/readyz")" = "200" ] && pass "readiness готова" || fail "readiness не готова: $(curl -s "$BASE/readyz")"

echo "-- контракт распознавания"
RESULT=$(curl -s -F "file=@$PDF" "$BASE/ocr?profile=balanced")
echo "$RESULT" | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert d["text"].strip(), "пустой текст"
for item in d["fields"]["bin"]:
    assert item["status"] in ("valid", "repaired", "unverified"), item
    assert item["requires_review"] is (item["status"] != "valid"), item
for item in d["fields"]["dates"]:
    assert item["status"] in ("valid", "invalid"), item
pages = len(d["pages"])
conf = d["mean_conf"]
warns = len(d["warnings"])
print("  страниц {}, уверенность {}, предупреждений {}".format(pages, conf, warns))
' && pass "поля отдаются со статусами" || fail "структура полей не соответствует контракту"

echo "-- границы входа"
head -c 17000000 /dev/urandom > /tmp/kzru-smoke-big.pdf
[ "$(code -F "file=@/tmp/kzru-smoke-big.pdf" "$BASE/ocr")" = "413" ] \
  && pass "файл больше лимита отклонён (413)" || fail "лимит размера не работает"

printf 'не pdf' > /tmp/kzru-smoke-bad.pdf
[ "$(code -F "file=@/tmp/kzru-smoke-bad.pdf" "$BASE/ocr")" = "422" ] \
  && pass "повреждённый PDF отклонён (422)" || fail "повреждённый PDF не даёт 422"

[ "$(code -F "file=@$PDF" "$BASE/ocr?profile=нет-такого")" = "400" ] \
  && pass "неизвестный профиль отклонён (400)" || fail "неизвестный профиль принят"

echo "-- реклама профилей совпадает с приёмом"
for p in $(curl -s "$BASE/readyz" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin)["profiles"]))'); do
  c=$(code -F "file=@$PDF" "$BASE/ocr?profile=$p")
  [ "$c" = "400" ] && fail "профиль $p рекламируется, но отвергнут"
done
pass "все рекламируемые профили принимаются"

echo "-- очередь заданий"
JOB=$(curl -s -F "file=@$PDF" "$BASE/jobs" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
for _ in $(seq 1 60); do
  STATUS=$(curl -s "$BASE/jobs/$JOB" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [ "$STATUS" != "running" ] && break
  sleep 2
done
[ "$STATUS" = "done" ] && pass "асинхронное задание выполнено" || fail "задание завершилось статусом $STATUS"

[ "$(code -X DELETE "$BASE/jobs/$JOB")" = "204" ] && pass "задание удаляется" || fail "удаление задания не работает"
[ "$(code "$BASE/jobs/$JOB")" = "404" ] && pass "удалённое задание недоступно" || fail "удалённое задание всё ещё отдаётся"

echo "-- журнал без утечек"
docker logs "$NAME" 2>&1 | grep -qi 'traceback' && fail "в журнале есть необработанные исключения" || pass "необработанных исключений нет"

rm -f /tmp/kzru-smoke-big.pdf /tmp/kzru-smoke-bad.pdf
echo "== дымовая проверка пройдена"
