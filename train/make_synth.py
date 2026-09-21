"""Генератор синтетических строк для дообучения Tesseract LSTM.

Формат вывода — ровно тот, который ждёт tesstrain: пара файлов на строку,
`<name>.png` и `<name>.gt.txt`, в одном каталоге.

Смысл дообучения: stock `kaz`/`rus` обучены на обобщённом наборе, а входящие
документы — печатные бланки РК в узком наборе гарнитур и с характерными
деградациями сканера. Подгонка модели под этот узкий домен дешева (обучение
идёт на CPU) и не требует человеческого времени, кроме запуска.

    python train/make_synth.py --out train/synth --count 30000 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

TEMPLATES_RU = [
    "Товарищество с ограниченной ответственностью «{org}»",
    "БИН {bin}, юридический адрес: город {city}, улица {street}, дом {house}",
    "Настоящий договор заключён {date} между сторонами на сумму {amount} ({amount_words}) тенге",
    "Директор {org} действует на основании устава, утверждённого {date}",
    "Справка выдана для представления в государственные органы Республики Казахстан",
    "Уставный капитал общества составляет {amount} ({amount_words}) тенге",
    "ИИН {bin}, дата рождения {date}, документ выдан органами юстиции",
    "Итого к оплате: {amount} тенге, в том числе налог на добавленную стоимость",
    "Приказ № {num} от {date} о назначении на должность главного бухгалтера",
    "Заявитель подтверждает достоверность представленных сведений и документов",
]

TEMPLATES_KK = [
    "«{org}» жауапкершілігі шектеулі серіктестігі",
    "БСН {bin}, заңды мекенжайы: {city} қаласы, {street} көшесі, {house} үй",
    "Осы шарт {date} күні тараптар арасында {amount} теңге сомасына жасалды",
    "Жарғы қоры {amount} ({amount_words}) теңгені құрайды",
    "Анықтама Қазақстан Республикасының мемлекеттік органдарына ұсыну үшін берілді",
    "Бұйрық № {num}, {date} күні бас бухгалтер қызметіне тағайындау туралы",
    "Өтініш беруші ұсынылған мәліметтердің дұрыстығын растайды",
    "Құрылтайшы шешімі {date} күні қабылданып, мөрмен куәландырылды",
]

ORGS = ["Астана Групп", "Қазақ Сервис", "Алатау Строй", "Нұр Логистик", "Темір Жол Сервис", "Ырыс Трейд"]
CITIES = ["Астана", "Алматы", "Шымкент", "Қарағанды", "Ақтөбе", "Өскемен", "Тараз"]
STREETS = ["Абая", "Әуезова", "Жібек жолы", "Республики", "Бөгенбай батыра", "Сәтпаева"]
UNITS = ["пятьсот тысяч", "один миллион двести пятьдесят тысяч", "триста сорок тысяч", "два миллиона"]


def _control_digit(first11: str) -> int | None:
    """Контрольный разряд БИН/ИИН РК (дубль недопустим — см. ocr/domain)."""
    digits = [int(c) for c in first11]
    s = sum((i + 1) * d for i, d in enumerate(digits)) % 11
    if s != 10:
        return s
    weights = (3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2)
    s2 = sum(w * d for w, d in zip(weights, digits)) % 11
    return None if s2 == 10 else s2


def _random_bin(rnd: random.Random) -> str:
    while True:
        head = f"{rnd.randint(0, 99):02d}{rnd.randint(1, 12):02d}{rnd.randint(4, 6)}{rnd.randint(0, 3)}"
        tail = "".join(str(rnd.randint(0, 9)) for _ in range(5))
        first11 = head + tail
        check = _control_digit(first11)
        if check is not None:
            return first11 + str(check)


def _random_amount(rnd: random.Random) -> tuple[str, str]:
    words = rnd.choice(UNITS)
    mapping = {
        "пятьсот тысяч": 500_000,
        "один миллион двести пятьдесят тысяч": 1_250_000,
        "триста сорок тысяч": 340_000,
        "два миллиона": 2_000_000,
    }
    value = mapping[words]
    return f"{value:,}".replace(",", " "), words


def make_text(rnd: random.Random) -> str:
    template = rnd.choice(TEMPLATES_RU + TEMPLATES_KK)
    amount, amount_words = _random_amount(rnd)
    return template.format(
        org=rnd.choice(ORGS),
        city=rnd.choice(CITIES),
        street=rnd.choice(STREETS),
        house=rnd.randint(1, 240),
        bin=_random_bin(rnd),
        date=f"{rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}.{rnd.randint(2015, 2026)}",
        amount=amount,
        amount_words=amount_words,
        num=rnd.randint(1, 999),
    )


def render_line(text: str, font_path: str, size: int) -> np.ndarray:
    font = ImageFont.truetype(font_path, size)
    dummy = Image.new("L", (1, 1), 255)
    box = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 24, box[3] - box[1] + 20
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((12 - box[0], 10 - box[1]), text, font=font, fill=0)
    return np.array(img)


def degrade(img: np.ndarray, rnd: random.Random) -> np.ndarray:
    angle = rnd.uniform(-1.5, 1.5)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    if rnd.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), rnd.uniform(0.3, 1.1))

    if rnd.random() < 0.4:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel) if rnd.random() < 0.5 else cv2.dilate(img, kernel)

    noise = rnd.uniform(2.0, 12.0)
    img = np.clip(img.astype(np.float32) + np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    quality = rnd.randint(35, 85)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Синтетические строки для tesstrain")
    parser.add_argument("--out", default="train/synth", help="каталог ground-truth")
    parser.add_argument("--count", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", help="файл с реальным текстом: по строке на строку")
    args = parser.parse_args(argv)

    fonts = [p for p in FONT_CANDIDATES if Path(p).exists()]
    if not fonts:
        parser.error(
            "не найдено ни одного TTF из списка. Установите fonts-dejavu-core и fonts-liberation "
            "или укажите свои пути в FONT_CANDIDATES"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(args.seed)
    np.random.seed(args.seed)

    corpus: list[str] = []
    if args.corpus:
        corpus = [ln.strip() for ln in Path(args.corpus).read_text(encoding="utf-8").splitlines() if ln.strip()]

    for i in range(args.count):
        text = rnd.choice(corpus) if corpus and rnd.random() < 0.5 else make_text(rnd)
        img = degrade(render_line(text, rnd.choice(fonts), rnd.randint(22, 38)), rnd)
        name = f"kzru_{i:06d}"
        cv2.imwrite(str(out / f"{name}.png"), img)
        (out / f"{name}.gt.txt").write_text(text + "\n", encoding="utf-8")

    print(f"готово: {args.count} строк в {out}, шрифтов: {len(fonts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
