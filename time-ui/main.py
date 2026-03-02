from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal

import pytz
from dateutil import parser as dateutil_parser

from settings import DEFAULT_LOCALE, ALLOWED_LOCALES, t

app = FastAPI(
    title="Secure Time Utilities API",
    version="1.0.0",
    description="Provides secure UTC/local time retrieval, formatting, timezone conversion, and comparison.",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Language negotiation (fallback: body/query -> Accept-Language -> default)
# -------------------------------
def resolve_lang(lang_value: str | None, request: Request) -> str:
    """Выбор локали: явный lang -> первый тег Accept-Language -> DEFAULT_LOCALE."""
    if lang_value is not None and lang_value != "":
        if lang_value not in ALLOWED_LOCALES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid lang. Allowed: {', '.join(ALLOWED_LOCALES)}",
            )
        return lang_value
    accept = request.headers.get("accept-language") or ""
    if accept:
        first_tag = accept.split(",")[0].strip().split(";")[0].strip()
        if first_tag in ALLOWED_LOCALES:
            return first_tag
    return DEFAULT_LOCALE


# -------------------------------
# Pydantic models
# -------------------------------
class FormatTimeInput(BaseModel):
    format: str = Field(
        "%Y-%m-%d %H:%M:%S", description="Python strftime format string"
    )
    timezone: str = Field(
        "UTC", description="IANA timezone name (e.g., UTC, America/New_York)"
    )
    lang: str | None = Field(
        default=None,
        description="Locale BCP 47 (optional). Validated against ALLOWED_LOCALES; default from Accept-Language or ru-RU.",
    )


class ConvertTimeInput(BaseModel):
    timestamp: str = Field(
        ..., description="ISO 8601 formatted time string (e.g., 2024-01-01T12:00:00Z)"
    )
    from_tz: str = Field(
        ..., description="Original IANA time zone of input (e.g. UTC or Europe/Berlin)"
    )
    to_tz: str = Field(..., description="Target IANA time zone to convert to")
    lang: str | None = Field(
        default=None,
        description="Locale BCP 47 (optional). Validated against ALLOWED_LOCALES; default from Accept-Language or ru-RU.",
    )


class ElapsedTimeInput(BaseModel):
    start: str = Field(..., description="Start timestamp in ISO 8601 format")
    end: str = Field(..., description="End timestamp in ISO 8601 format")
    units: Literal["seconds", "minutes", "hours", "days"] = Field(
        "seconds", description="Unit for elapsed time"
    )
    lang: str | None = Field(
        default=None,
        description="Locale BCP 47 (optional). Validated against ALLOWED_LOCALES; default from Accept-Language or ru-RU.",
    )


class ParseTimestampInput(BaseModel):
    timestamp: str = Field(
        ..., description="Flexible input timestamp string (e.g., 2024-06-01 12:00 PM)"
    )
    timezone: str = Field(
        "UTC", description="Assumed timezone if none is specified in input"
    )
    lang: str | None = Field(
        default=None,
        description="Locale BCP 47 (optional). Validated against ALLOWED_LOCALES; default from Accept-Language or ru-RU.",
    )


# -------------------------------
# Helper for pretty HTML
# -------------------------------
def html_page(title: str, body: str) -> HTMLResponse:
    html = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f7f7f7;
                height: 4rem;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
            }}
            pre {{
                background: #ecf0f1;
                padding: 10px;
                border-radius: 5px;
                font-size: 1.1em;
            }}
            ul {{
                columns: 3;
                -webkit-columns: 3;
                -moz-columns: 3;
            }}
            li {{
                margin-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {body}
    </body>
    </html>
    """
    return html


# -------------------------------
# Routes
# -------------------------------
@app.get("/get_current_local_time", summary="Current Local Time")
def get_current_local(
    request: Request,
    lang: str | None = Query(default=None, description="Locale BCP 47 (optional)."),
):
    locale = resolve_lang(lang, request)
    local_time = datetime.now().isoformat()
    headers = {"Content-Disposition": "inline"}
    return HTMLResponse(
        content=html_page(t(locale, "page_title_current_local_time"), f"<pre>{local_time}</pre>"),
        headers=headers,
    )


@app.post("/format_time", summary="Format current time")
def format_current_time(request: Request, data: FormatTimeInput):
    locale = resolve_lang(data.lang, request)
    try:
        tz = pytz.timezone(data.timezone)
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"Invalid timezone: {data.timezone}"
        )
    now = datetime.now(tz)
    try:
        formatted = now.strftime(data.format)
        headers = {"Content-Disposition": "inline"}
        return HTMLResponse(
            content=html_page(t(locale, "page_title_formatted_time"), f"<pre>{formatted}</pre>"),
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format string: {e}")


@app.post("/convert_time", summary="Convert between timezones")
def convert_time(request: Request, data: ConvertTimeInput):
    locale = resolve_lang(data.lang, request)
    try:
        from_zone = pytz.timezone(data.from_tz)
        to_zone = pytz.timezone(data.to_tz)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid timezone: {e}")
    try:
        dt = dateutil_parser.parse(data.timestamp)
        if dt.tzinfo is None:
            dt = from_zone.localize(dt)
        else:
            dt = dt.astimezone(from_zone)
        converted = dt.astimezone(to_zone).isoformat()
        headers = {"Content-Disposition": "inline"}
        return HTMLResponse(
            content=html_page(t(locale, "page_title_converted_time"), f"<pre>{converted}</pre>"),
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {e}")


@app.post("/parse_timestamp", summary="Parse and normalize timestamps")
def parse_timestamp(request: Request, data: ParseTimestampInput):
    locale = resolve_lang(data.lang, request)
    try:
        tz = pytz.timezone(data.timezone)
        dt = dateutil_parser.parse(data.timestamp)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        dt_utc = dt.astimezone(pytz.utc).isoformat()
        headers = {"Content-Disposition": "inline"}
        return HTMLResponse(
            content=html_page(t(locale, "page_title_parsed_timestamp_utc"), f"<pre>{dt_utc}</pre>"),
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse: {e}")


@app.post("/elapsed_time", summary="Elapsed time between two timestamps")
def elapsed_time(request: Request, data: ElapsedTimeInput):
    locale = resolve_lang(data.lang, request)
    try:
        start_dt = dateutil_parser.parse(data.start)
        end_dt = dateutil_parser.parse(data.end)
        if start_dt.tzinfo is None:
            start_dt = pytz.utc.localize(start_dt)
        if end_dt.tzinfo is None:
            end_dt = pytz.utc.localize(end_dt)
        delta_seconds = (end_dt - start_dt).total_seconds()
        if delta_seconds < 0:
            raise HTTPException(
                status_code=400,
                detail="End timestamp must be after start timestamp",
            )
        units = data.units
        if units == "seconds":
            value = delta_seconds
        elif units == "minutes":
            value = delta_seconds / 60
        elif units == "hours":
            value = delta_seconds / 3600
        else:  # days
            value = delta_seconds / 86400
        unit_key = f"unit_{units}"
        result = f"{value} {t(locale, unit_key)}"
        headers = {"Content-Disposition": "inline"}
        return HTMLResponse(
            content=html_page(t(locale, "page_title_elapsed_time"), f"<pre>{result}</pre>"),
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamps: {e}")


@app.get("/list_time_zones", summary="All valid time zones")
def list_time_zones(
    request: Request,
    lang: str | None = Query(default=None, description="Locale BCP 47 (optional)."),
):
    locale = resolve_lang(lang, request)
    tz_list_html = (
        "<ul>" + "".join(f"<li>{tz}</li>" for tz in pytz.all_timezones) + "</ul>"
    )
    headers = {"Content-Disposition": "inline"}
    return HTMLResponse(
        content=html_page(t(locale, "page_title_valid_time_zones"), tz_list_html), headers=headers
    )


@app.get("/go_to_timezones", summary="Redirect to /list_time_zones")
def redirect_to_timezones(
    request: Request,
    lang: str | None = Query(default=None, description="Locale BCP 47 (optional)."),
):
    resolve_lang(lang, request)
    return RedirectResponse(url="/list_time_zones")


@app.get("/useful_redirect", summary="Redirect to useful external time resource")
def useful_redirect(
    request: Request,
    lang: str | None = Query(default=None, description="Locale BCP 47 (optional)."),
):
    resolve_lang(lang, request)
    headers = {"Content-Disposition": "inline"}
    external_url = "https://time.is"
    return RedirectResponse(url=external_url, headers=headers)
