"""Offline stand-ins for data.binance.vision: synthetic klines, archives and a fake bucket."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from xml.sax.saxutils import escape

import httpx

from olsprime.data.binance_vision import Month, archive_relpath

HOUR_MS = 3_600_000
DAY_MS = 86_400_000

CSV_COLUMNS = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,"
    "count,taker_buy_volume,taker_buy_quote_volume,ignore"
)


def epoch_ms(*args: int) -> int:
    """Epoch milliseconds of ``datetime(*args)`` in UTC."""
    return int(datetime(*args, tzinfo=UTC).timestamp()) * 1000


def kline_rows(
    start_ms: int,
    count: int,
    step_ms: int,
    *,
    unit: str = "ms",
    price: float = 100.0,
    quote_volume: float = 1_000.0,
) -> list[list[str]]:
    """``count`` consistent bars from ``start_ms``, with timestamps written in ``unit`` (ms or us)."""
    scale = {"ms": 1, "us": 1_000}[unit]
    rows = []
    for i in range(count):
        open_ms = start_ms + i * step_ms
        o = price + i
        rows.append(
            [
                str(open_ms * scale),
                f"{o:.8f}",
                f"{o + 2:.8f}",
                f"{o - 1:.8f}",
                f"{o + 1:.8f}",
                "10.00000000",
                str((open_ms + step_ms) * scale - 1),
                f"{quote_volume:.8f}",
                "42",
                "5.00000000",
                f"{quote_volume / 2:.8f}",
                "0",
            ]
        )
    return rows


def month_rows(month: Month, step_ms: int, **kwargs) -> list[list[str]]:
    """Every bar of ``month``."""
    start = epoch_ms(month.year, month.month, 1)
    nxt = month.shift(1)
    count = (epoch_ms(nxt.year, nxt.month, 1) - start) // step_ms
    return kline_rows(start, count, step_ms, **kwargs)


def klines_csv(rows: Iterable[list[str]], *, header: bool = False) -> bytes:
    lines = [",".join(row) for row in rows]
    if header:
        lines.insert(0, CSV_COLUMNS)
    return ("\n".join(lines) + "\n").encode()


def zip_archive(member: str, data: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, data)
    return buffer.getvalue()


class FakeBucket:
    """In-memory data.binance.vision: file downloads plus the S3 ListObjects (v1) API."""

    def __init__(self, page_size: int = 1000) -> None:
        self.objects: dict[str, bytes] = {}  # bucket key -> body
        self.requests: list[httpx.URL] = []
        self.page_size = page_size

    def add_month(
        self,
        symbol: str,
        interval: str,
        month: Month,
        rows: list[list[str]],
        *,
        bad_checksum: bool = False,
    ) -> None:
        name = f"{symbol}-{interval}-{month}"
        archive = zip_archive(f"{name}.csv", klines_csv(rows))
        digest = hashlib.sha256(b"something else" if bad_checksum else archive).hexdigest()
        key = f"data/{archive_relpath(symbol, interval, month)}"
        self.objects[key] = archive
        self.objects[key + ".CHECKSUM"] = f"{digest}  {name}.zip\n".encode()

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url)
        if request.url.host == "s3-ap-northeast-1.amazonaws.com":
            return self._list(request.url.params)
        key = request.url.path.lstrip("/")
        if request.url.host == "data.binance.vision" and key in self.objects:
            return httpx.Response(200, content=self.objects[key])
        return httpx.Response(404)

    def _list(self, params: httpx.QueryParams) -> httpx.Response:
        prefix, delimiter, marker = (params.get(k, "") for k in ("prefix", "delimiter", "marker"))
        entries: dict[str, bool] = {}  # name -> is a common prefix
        for key in self.objects:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                entries[prefix + rest.split(delimiter, 1)[0] + delimiter] = True
            else:
                entries[key] = False
        names = sorted(name for name in entries if name > marker)
        page, truncated = names[: self.page_size], len(names) > self.page_size
        xml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
            f"<Name>data.binance.vision</Name><Prefix>{escape(prefix)}</Prefix>",
            f"<Marker>{escape(marker)}</Marker>",
        ]
        if truncated:
            xml.append(f"<NextMarker>{escape(page[-1])}</NextMarker>")
        xml.append(f"<IsTruncated>{str(truncated).lower()}</IsTruncated>")
        for name in page:
            if entries[name]:
                xml.append(f"<CommonPrefixes><Prefix>{escape(name)}</Prefix></CommonPrefixes>")
            else:
                xml.append(f"<Contents><Key>{escape(name)}</Key><Size>{len(self.objects[name])}</Size></Contents>")
        xml.append("</ListBucketResult>")
        return httpx.Response(200, content="".join(xml).encode())
