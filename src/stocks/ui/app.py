from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from stocks.ui.service import ViewModelStore


PAGE_ROUTES = {
    "/": ("dashboard.html", "Operations"),
    "/signals": ("signals.html", "Trending Signals"),
    "/universe": ("universe.html", "Universe Explorer"),
    "/sectors": ("dimension.html", "Sector Map"),
    "/industries": ("dimension.html", "Industry Map"),
    "/regions": ("dimension.html", "Regional Map"),
    "/etfs": ("instruments.html", "ETF Explorer"),
    "/commodities": ("instruments.html", "Commodity Dashboard"),
    "/strategies": ("strategies.html", "Strategies"),
    "/portfolio": ("portfolio.html", "Portfolio"),
    "/performance": ("performance.html", "P&L Calendar"),
    "/news": ("news.html", "Market Intelligence"),
    "/research": ("research.html", "Research"),
    "/health": ("health.html", "System Health"),
    "/audit": ("audit.html", "Audit"),
}


def create_app(project_root: Path) -> FastAPI:
    package_root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=package_root / "templates")
    store = ViewModelStore(project_root)
    app = FastAPI(
        title="Stocks Operations Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.project_root = project_root
    app.state.store = store
    app.mount(
        "/static",
        StaticFiles(directory=package_root / "static"),
        name="static",
    )
    analytics_root = project_root / "output" / "ui" / "analytics"
    analytics_root.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/analytics",
        StaticFiles(directory=analytics_root),
        name="analytics",
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Any
    ) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'self'"
        )
        return response

    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return JSONResponse(store.health())

    @app.get("/api/dashboard")
    async def api_dashboard() -> JSONResponse:
        return JSONResponse(store.dashboard())

    @app.get("/api/signals")
    async def api_signals(
        collection: str = "diversified",
        instrument_type: str | None = None,
        sector: str | None = None,
        region: str | None = None,
        timeframe: str | None = None,
        status: str | None = None,
        minimum_score: float = Query(0.0, ge=0.0, le=1.0),
    ) -> JSONResponse:
        return JSONResponse(
            store.signals(
                collection=collection,
                instrument_type=instrument_type,
                sector=sector,
                region=region,
                timeframe=timeframe,
                status=status,
                minimum_score=minimum_score,
            )
        )

    @app.get("/api/universe")
    async def api_universe(
        query: str = "",
        instrument_type: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        region: str | None = None,
        eligibility: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=10, le=100),
        sort: str = "symbol",
    ) -> JSONResponse:
        return JSONResponse(
            store.universe(
                query=query,
                instrument_type=instrument_type,
                sector=sector,
                industry=industry,
                region=region,
                eligibility=eligibility,
                page=page,
                page_size=page_size,
                sort=sort,
            )
        )

    @app.get("/api/dimensions/{dimension}")
    async def api_dimension(dimension: str) -> JSONResponse:
        payload = store.dimension(dimension)
        return JSONResponse(
            payload,
            status_code=200
            if payload.get("status") != "BLOCKED"
            else 404,
        )

    @app.get("/api/strategies")
    async def api_strategies() -> JSONResponse:
        return JSONResponse(store.strategies())

    @app.get("/api/portfolio")
    async def api_portfolio() -> JSONResponse:
        return JSONResponse(store.portfolio())

    @app.get("/api/performance")
    async def api_performance(
        month: str | None = None,
        environment: str | None = None,
    ) -> JSONResponse:
        return JSONResponse(store.performance(month, environment))

    @app.get("/api/research")
    async def api_research() -> JSONResponse:
        return JSONResponse(store.research())

    @app.get("/api/audit")
    async def api_audit() -> JSONResponse:
        return JSONResponse(store.audit())

    @app.get("/api/analysis/coverage")
    async def api_analysis_coverage() -> JSONResponse:
        return JSONResponse(store.analysis_coverage())

    @app.get("/api/asset/{symbol}")
    async def api_asset(symbol: str) -> JSONResponse:
        payload = store.asset(symbol)
        return JSONResponse(
            payload,
            status_code=200
            if payload.get("status")
            not in {"BLOCKED", "NOT_IN_UNIVERSE"}
            else 404,
        )

    @app.get("/api/news")
    async def api_news() -> JSONResponse:
        return JSONResponse(store.news())

    @app.get("/api/chart/{symbol}")
    async def api_chart(
        symbol: str,
        interval: str = "1d",
        limit: int = Query(300, ge=50, le=1000),
    ) -> JSONResponse:
        payload = store.chart(symbol, interval, limit)
        return JSONResponse(
            payload,
            status_code=200
            if payload.get("status") != "BLOCKED"
            else 400,
        )

    @app.get("/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            previous = ""
            while not await request.is_disconnected():
                fingerprint = store.event_fingerprint()
                if fingerprint != previous:
                    event = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "fingerprint": fingerprint,
                        "dashboard": store.dashboard(),
                    }
                    yield (
                        "event: state\n"
                        f"data: {json.dumps(event, default=str)}\n\n"
                    )
                    previous = fingerprint
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    def render(
        request: Request,
        template: str,
        title: str,
        viewmodel: dict[str, Any],
        **extra: Any,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name=template,
            context={
                "title": title,
                "active_path": request.url.path,
                "viewmodel": viewmodel,
                **extra,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/"][0],
            PAGE_ROUTES["/"][1],
            store.dashboard(),
        )

    @app.get("/signals", response_class=HTMLResponse)
    async def signals_page(
        request: Request,
        collection: str = "diversified",
        minimum_score: float = Query(0.0, ge=0.0, le=1.0),
    ) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/signals"][0],
            PAGE_ROUTES["/signals"][1],
            store.signals(
                collection=collection,
                minimum_score=minimum_score,
            ),
            collection=collection,
            minimum_score=minimum_score,
        )

    @app.get("/universe", response_class=HTMLResponse)
    async def universe_page(
        request: Request,
        query: str = "",
        instrument_type: str | None = None,
        sector: str | None = None,
        region: str | None = None,
        page: int = Query(1, ge=1),
    ) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/universe"][0],
            PAGE_ROUTES["/universe"][1],
            store.universe(
                query=query,
                instrument_type=instrument_type,
                sector=sector,
                region=region,
                page=page,
            ),
        )

    for route, dimension in (
        ("/sectors", "sector"),
        ("/industries", "industry"),
        ("/regions", "region"),
    ):

        async def dimension_page(
            request: Request,
            dimension_name: str = dimension,
            route_name: str = route,
        ) -> HTMLResponse:
            return render(
                request,
                PAGE_ROUTES[route_name][0],
                PAGE_ROUTES[route_name][1],
                store.dimension(dimension_name),
                dimension=dimension_name,
            )

        app.add_api_route(
            route,
            dimension_page,
            methods=["GET"],
            response_class=HTMLResponse,
            name=f"{dimension}_page",
        )

    @app.get("/etfs", response_class=HTMLResponse)
    async def etfs_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/etfs"][0],
            PAGE_ROUTES["/etfs"][1],
            store.instruments_by_type("ETF"),
            instrument_kind="ETF",
        )

    @app.get("/commodities", response_class=HTMLResponse)
    async def commodities_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/commodities"][0],
            PAGE_ROUTES["/commodities"][1],
            store.instruments_by_type("COMMODITY_EXPOSURE"),
            instrument_kind="Commodity exposure",
        )

    @app.get("/strategies", response_class=HTMLResponse)
    async def strategies_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/strategies"][0],
            PAGE_ROUTES["/strategies"][1],
            store.strategies(),
        )

    @app.get("/portfolio", response_class=HTMLResponse)
    async def portfolio_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/portfolio"][0],
            PAGE_ROUTES["/portfolio"][1],
            store.portfolio(),
        )

    @app.get("/performance", response_class=HTMLResponse)
    async def performance_page(
        request: Request,
        month: str | None = None,
        environment: str | None = None,
    ) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/performance"][0],
            PAGE_ROUTES["/performance"][1],
            store.performance(month, environment),
        )

    @app.get("/news", response_class=HTMLResponse)
    async def news_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/news"][0],
            PAGE_ROUTES["/news"][1],
            store.news(),
        )

    @app.get("/asset/{symbol}", response_class=HTMLResponse)
    async def asset_page(
        request: Request,
        symbol: str,
    ) -> HTMLResponse:
        payload = store.asset(symbol)
        return render(
            request,
            "asset.html",
            f"{str(symbol).upper()} Analysis",
            payload,
        )

    @app.get("/research", response_class=HTMLResponse)
    async def research_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/research"][0],
            PAGE_ROUTES["/research"][1],
            store.research(),
        )

    @app.get("/health", response_class=HTMLResponse)
    async def health_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/health"][0],
            PAGE_ROUTES["/health"][1],
            store.health(),
        )

    @app.get("/audit", response_class=HTMLResponse)
    async def audit_page(request: Request) -> HTMLResponse:
        return render(
            request,
            PAGE_ROUTES["/audit"][0],
            PAGE_ROUTES["/audit"][1],
            store.audit(),
        )

    return app
