"""Crimetor landing server: static pages plus Stripe Checkout for pricing.html.

Run from the landing folder:
    python -m pip install -r requirements.txt
    copy .env.example .env   (then put your Stripe test keys into .env)
    python server.py
and open http://localhost:8000/pricing.html
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

LANDING_DIR = Path(__file__).resolve().parent
load_dotenv(LANDING_DIR / ".env")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
DB_PATH = LANDING_DIR / "payments.sqlite3"

# Must match the plans in pricing.html. The server is the source of truth for amounts.
CURRENCY = "kzt"  # KZT is a two-decimal currency in Stripe: amounts are sent in tiyn.
ANNUAL_DISCOUNT = 0.15
PLANS = {
    "start": {"title": "Start", "monthly": 29_900, "cameras": 4},
    "campus": {"title": "Campus", "monthly": 99_900, "cameras": 16},
    "network": {"title": "Network", "monthly": 249_900, "cameras": 48},
}
PERIOD_TITLES = {"month": "1 месяц", "year": "1 год, предоплата -15%"}

STATIC_NAME = re.compile(r"^[A-Za-z0-9_-]+\.(html|css|js)$")
SESSION_ID = re.compile(r"^cs_(test|live)_[A-Za-z0-9]+$")

log = logging.getLogger("crimetor.billing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

stripe_client = stripe.StripeClient(STRIPE_SECRET_KEY) if STRIPE_SECRET_KEY else None


def plan_amount(plan: str, period: str) -> int:
    """Amount in whole tenge, same formula as quoteAmount() in pricing.html."""
    monthly = PLANS[plan]["monthly"]
    return round(monthly * 12 * (1 - ANNUAL_DISCOUNT)) if period == "year" else monthly


class CheckoutRequest(BaseModel):
    plan: Literal["start", "campus", "network"]
    period: Literal["month", "year"]
    cameras: int = Field(ge=1)
    organization: str = Field(max_length=120)
    email: str = Field(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    @field_validator("organization")
    @classmethod
    def organization_not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Введите название организации")
        return value

    @model_validator(mode="after")
    def cameras_within_plan(self) -> "CheckoutRequest":
        if self.cameras > PLANS[self.plan]["cameras"]:
            raise ValueError(f"В тарифе {PLANS[self.plan]['title']} не больше {PLANS[self.plan]['cameras']} камер")
        return self


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS payments (
                session_id TEXT PRIMARY KEY,
                plan TEXT NOT NULL,
                period TEXT NOT NULL,
                cameras INTEGER NOT NULL,
                organization TEXT NOT NULL,
                email TEXT,
                amount_total INTEGER NOT NULL,
                currency TEXT NOT NULL,
                payment_intent TEXT,
                paid_at TEXT NOT NULL
            )"""
        )


def record_payment(session) -> bool:
    """Store a paid Checkout Session once. Returns False if it was already stored."""
    meta = session.metadata
    with sqlite3.connect(DB_PATH) as db:
        cur = db.execute(
            "INSERT OR IGNORE INTO payments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                meta["plan"],
                meta["period"],
                int(meta["cameras"]),
                meta["organization"],
                session.customer_details.email if session.customer_details else session.customer_email,
                session.amount_total,
                session.currency,
                session.payment_intent if isinstance(session.payment_intent, str) else None,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return cur.rowcount == 1


def require_stripe() -> stripe.StripeClient:
    if stripe_client is None:
        raise HTTPException(503, "Оплата не настроена: укажите STRIPE_SECRET_KEY в landing/.env и перезапустите сервер.")
    return stripe_client


app = FastAPI(title="Crimetor landing", docs_url=None, redoc_url=None, openapi_url=None)
init_db()

FIELD_TITLES = {
    "plan": "тариф",
    "period": "оплата",
    "cameras": "количество камер",
    "organization": "организация",
    "email": "рабочая почта",
}


@app.exception_handler(RequestValidationError)
async def readable_validation_error(request: Request, exc: RequestValidationError):
    """Turn pydantic errors into one Russian sentence that pricing.html can show as is."""
    problems = []
    for error in exc.errors():
        field = next((FIELD_TITLES[part] for part in error["loc"] if part in FIELD_TITLES), None)
        message = str(error.get("msg", "")).removeprefix("Value error, ")
        problems.append(f"{field}: проверьте значение" if field else message or "неверные данные")
    log.info("Checkout rejected: %s", exc.errors())
    return JSONResponse(status_code=422, content={"detail": "Проверьте поля формы. " + "; ".join(problems) + "."})


@app.post("/api/checkout")
def create_checkout(body: CheckoutRequest, request: Request):
    client = require_stripe()
    plan = PLANS[body.plan]
    amount = plan_amount(body.plan, body.period)
    base_url = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    metadata = {
        "plan": body.plan,
        "period": body.period,
        "cameras": str(body.cameras),
        "organization": body.organization,
    }
    try:
        session = client.v1.checkout.sessions.create(
            params={
                "mode": "payment",
                "locale": "ru",
                "customer_email": body.email,
                "line_items": [
                    {
                        "quantity": 1,
                        "price_data": {
                            "currency": CURRENCY,
                            "unit_amount": amount * 100,
                            "product_data": {
                                "name": f"Crimetor {plan['title']}, {PERIOD_TITLES[body.period]}",
                                "description": f"{body.organization}. Камер: {body.cameras} из {plan['cameras']}.",
                            },
                        },
                    }
                ],
                "metadata": metadata,
                "payment_intent_data": {"metadata": metadata},
                "success_url": f"{base_url}/pricing.html?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
                "cancel_url": f"{base_url}/pricing.html?checkout=cancel",
            }
        )
    except stripe.StripeError as exc:
        log.warning("Stripe checkout create failed: %s", exc.user_message or exc)
        raise HTTPException(502, exc.user_message or "Stripe не принял запрос. Попробуйте позже.") from exc
    log.info("Checkout session %s: %s/%s, %s KZT", session.id, body.plan, body.period, amount)
    return {"url": session.url}


@app.get("/api/checkout/{session_id}")
def checkout_status(session_id: str):
    """Shown to the buyer after returning from Stripe. Access is granted only by the webhook."""
    if not SESSION_ID.match(session_id):
        raise HTTPException(404, "Оплата не найдена")
    client = require_stripe()
    try:
        session = client.v1.checkout.sessions.retrieve(session_id)
    except stripe.InvalidRequestError as exc:
        raise HTTPException(404, "Оплата не найдена") from exc
    except stripe.StripeError as exc:
        raise HTTPException(502, "Не удалось получить статус оплаты. Обновите страницу позже.") from exc
    meta = session.metadata
    return {
        "status": session.status,
        "payment_status": session.payment_status,
        "plan": PLANS.get(meta["plan"], {}).get("title", meta["plan"]) if "plan" in meta else None,
        "period": meta["period"] if "period" in meta else None,
        "amount": session.amount_total // 100 if session.amount_total is not None else None,
        "currency": session.currency,
    }


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    client = require_stripe()
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, "STRIPE_WEBHOOK_SECRET is not set")
    payload = await request.body()
    try:
        event = client.construct_event(payload, request.headers.get("stripe-signature"), STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.SignatureVerificationError) as exc:
        raise HTTPException(400, "Invalid signature") from exc

    if event.type not in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        return {"received": True}
    session = event.data.object
    if session.payment_status != "paid":
        log.info("Checkout session %s completed, payment pending", session.id)
        return {"received": True}

    meta = session.metadata
    plan, period = meta["plan"] if "plan" in meta else None, meta["period"] if "period" in meta else None
    if plan not in PLANS or period not in PERIOD_TITLES or "cameras" not in meta or "organization" not in meta:
        log.error("Checkout session %s has unexpected metadata, not recorded", session.id)
        return {"received": True}
    expected = plan_amount(plan, period) * 100
    if session.amount_total != expected or session.currency != CURRENCY:
        log.error(
            "Checkout session %s amount mismatch: got %s %s, expected %s %s",
            session.id, session.amount_total, session.currency, expected, CURRENCY,
        )
        return {"received": True}

    if await run_in_threadpool(record_payment, session):
        log.info("Payment recorded: %s, %s %s for %s", session.id, plan, period, meta["organization"])
    return {"received": True}


@app.get("/")
def index():
    return FileResponse(LANDING_DIR / "index.html")


@app.get("/{name}")
def static_page(name: str):
    # Only page assets are served, never .env, the database or this file.
    path = LANDING_DIR / name
    if not STATIC_NAME.match(name) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
