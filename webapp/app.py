"""
webapp/app.py

Flask application factory for the Value Betting Engine web UI.

Design notes
------------
* One shared :class:`~webapp.store.Store` (SQLite) holds the risk profile
  and bet ledger.
* Live slates are scored on demand through :func:`core.scoring_engine.score_slate`
  and cached in-process for ``SLATE_TTL`` seconds — every score costs one
  odds-provider API call, so a refresh within the window is served from
  cache rather than burning quota.
* Every route degrades gracefully: a sport with no trained model, no live
  games, or no edges renders an explanatory panel rather than an error.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from flask import (
    Flask, abort, flash, redirect, render_template, request, url_for,
)
from markupsafe import Markup

from config.settings import UserRiskProfile
from core.parlay import ParlayLeg, analyze_parlay
from core.scoring_engine import SlateResult, score_slate
from risk_bridge.risk_manager import RiskManager
from sports import all_plugins, get_plugin
from webapp.risk_metrics import bankroll_curve, compute_metrics
from webapp.store import Store

logger = logging.getLogger(__name__)

SLATE_TTL = 60  # seconds


def create_app(store: Optional[Store] = None) -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "value-betting-engine-local"  # local UI only; flash msgs
    app.jinja_env.globals["current_year"] = time.strftime("%Y")
    app.jinja_env.globals["chart_svg"] = _chart_svg

    store = store or Store()
    _slate_cache: Dict[str, Tuple[float, SlateResult]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _risk_manager() -> RiskManager:
        p = store.get_profile()
        profile = UserRiskProfile(
            liquid_bankroll=p["liquid_bankroll"],
            disposable_income=p["disposable_income"],
            volatility_tolerance=int(p["volatility_tolerance"]),
        )
        return RiskManager(profile)

    def _get_slate(sport_key: str, force: bool = False) -> SlateResult:
        now = time.time()
        cached = _slate_cache.get(sport_key)
        if cached and not force and (now - cached[0]) < SLATE_TTL:
            return cached[1]
        result = score_slate(sport_key, _risk_manager())
        _slate_cache[sport_key] = (now, result)
        return result

    @app.context_processor
    def _inject_nav():
        return {
            "nav_sports": [
                {"key": p.config.key, "name": p.config.display_name}
                for p in all_plugins()
            ]
        }

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    @app.route("/")
    def dashboard():
        profile = store.get_profile()
        metrics = compute_metrics(store.settled_bets_chronological())
        pending = store.list_bets(status="pending")
        sports = [
            {
                "key": p.config.key,
                "name": p.config.display_name,
                "score_label": p.config.score_label,
                "market": "3-way (1X2)" if p.config.has_draws else "Moneyline",
                "trained": p.is_trained(),
            }
            for p in all_plugins()
        ]
        return render_template(
            "dashboard.html",
            profile=profile,
            metrics=metrics,
            sports=sports,
            pending_count=len(pending),
        )

    # ------------------------------------------------------------------
    # Per-sport slate
    # ------------------------------------------------------------------

    @app.route("/slate/<sport_key>")
    def slate(sport_key: str):
        try:
            get_plugin(sport_key)
        except KeyError:
            abort(404)
        force = request.args.get("refresh") == "1"
        result = _get_slate(sport_key, force=force)

        bets = result.bets
        # Best (highest-edge) row per game for a clean board view.
        board = []
        if not bets.empty:
            for gid, grp in bets.groupby("game_id"):
                grp = grp.sort_values("edge", ascending=False)
                board.append({
                    "game_id": gid,
                    "matchup": grp.iloc[0]["matchup"],
                    "commence_time": grp.iloc[0]["commence_time"],
                    "outcomes": grp.to_dict("records"),
                })
            board.sort(key=lambda g: g["commence_time"])

        allocation = (
            result.allocation.to_dict("records")
            if not result.allocation.empty else []
        )
        arbitrage = result.arbitrage.to_dict("records") if not result.arbitrage.empty else []

        return render_template(
            "slate.html",
            result=result,
            board=board,
            allocation=allocation,
            arbitrage=arbitrage,
        )

    # ------------------------------------------------------------------
    # Place / settle / delete bets
    # ------------------------------------------------------------------

    @app.route("/bet/place", methods=["POST"])
    def place_bet():
        f = request.form
        try:
            bet_id = store.add_bet(
                sport=f["sport"],
                matchup=f["matchup"],
                selection=f["selection"],
                price=float(f["price"]),
                stake=float(f["stake"]),
                bookmaker=f.get("bookmaker", ""),
                model_prob=float(f["model_prob"]) if f.get("model_prob") else None,
                edge=float(f["edge"]) if f.get("edge") else None,
            )
            flash(f"Bet #{bet_id} recorded: {f['selection']} (${float(f['stake']):.2f}).", "success")
        except (KeyError, ValueError) as exc:
            flash(f"Could not record bet: {exc}", "error")
        return redirect(request.referrer or url_for("dashboard"))

    @app.route("/bet/<int:bet_id>/settle", methods=["POST"])
    def settle_bet(bet_id: int):
        status = request.form.get("status", "")
        try:
            store.settle_bet(bet_id, status)
            flash(f"Bet #{bet_id} settled as {status}.", "success")
        except (ValueError, KeyError) as exc:
            flash(f"Could not settle bet: {exc}", "error")
        return redirect(url_for("history"))

    @app.route("/bet/<int:bet_id>/delete", methods=["POST"])
    def delete_bet(bet_id: int):
        store.delete_bet(bet_id)
        flash(f"Bet #{bet_id} deleted.", "success")
        return redirect(url_for("history"))

    # ------------------------------------------------------------------
    # History + risk analytics
    # ------------------------------------------------------------------

    @app.route("/history")
    def history():
        all_bets = store.list_bets()
        settled = store.settled_bets_chronological()
        metrics = compute_metrics(settled)
        curve = bankroll_curve(settled)
        return render_template(
            "history.html",
            bets=all_bets,
            metrics=metrics,
            curve=curve,
        )

    # ------------------------------------------------------------------
    # Parlay analyzer
    # ------------------------------------------------------------------

    @app.route("/parlay", methods=["GET", "POST"])
    def parlay():
        analysis = None
        legs_input = []
        if request.method == "POST":
            labels = request.form.getlist("label")
            probs = request.form.getlist("prob")
            odds = request.form.getlist("decimal_odds")
            games = request.form.getlist("game_id")
            for i, label in enumerate(labels):
                if not label.strip():
                    continue
                try:
                    legs_input.append(ParlayLeg(
                        label=label.strip(),
                        game_id=(games[i].strip() if i < len(games) and games[i].strip()
                                 else f"game_{i}"),
                        true_prob=float(probs[i]),
                        decimal_odds=float(odds[i]),
                    ))
                except (ValueError, IndexError):
                    continue
            if len(legs_input) >= 2:
                analysis = analyze_parlay(legs_input)
            else:
                flash("Add at least two valid legs (probability 0-1, decimal odds > 1).", "error")
        return render_template("parlay.html", analysis=analysis, legs=legs_input)

    # ------------------------------------------------------------------
    # Settings (risk profile)
    # ------------------------------------------------------------------

    @app.route("/settings", methods=["GET", "POST"])
    def settings():
        if request.method == "POST":
            try:
                bankroll = float(request.form["liquid_bankroll"])
                disposable = float(request.form["disposable_income"])
                tolerance = int(request.form["volatility_tolerance"])
                if bankroll < 0 or disposable < 0:
                    raise ValueError("amounts must be non-negative")
                if not 1 <= tolerance <= 10:
                    raise ValueError("tolerance must be 1-10")
                store.save_profile(bankroll, disposable, tolerance)
                _slate_cache.clear()  # allocations depend on the profile
                flash("Risk profile saved.", "success")
                return redirect(url_for("settings"))
            except (KeyError, ValueError) as exc:
                flash(f"Invalid input: {exc}", "error")

        profile = store.get_profile()
        # Preview the constraints this profile produces.
        rm = _risk_manager()
        max_port, max_single, risk_scaling, eff = rm.get_portfolio_constraints()
        preview = {
            "effective_bankroll": eff,
            "max_portfolio_pct": max_port * 100,
            "max_single_pct": max_single * 100,
            "max_drawdown_pct": rm.profile.max_drawdown_pct * 100,
            "session_stop_pct": rm.session_stop_loss_pct * 100,
        }
        return render_template("settings.html", profile=profile, preview=preview)

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404,
                               message="Page not found."), 404

    return app


def _chart_svg(curve: List[Dict], width: int = 640, height: int = 220) -> Markup:
    """
    Render a cumulative-P&L line chart as inline, dependency-free SVG.

    Server-side SVG keeps the page fully functional without JavaScript and
    lets the chart inherit the theme's colours via ``currentColor``. The
    parent element supplies an ``aria-label`` describing the trend, so this
    is decorative (``aria-hidden``) at the SVG level.
    """
    pad = 28
    pnls = [p["pnl"] for p in curve]
    n = len(curve)
    if n < 2:
        return Markup("")

    lo, hi = min(pnls), max(pnls)
    span = (hi - lo) or 1.0
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def x(i: int) -> float:
        return pad + plot_w * i / (n - 1)

    def y(v: float) -> float:
        return pad + plot_h * (1 - (v - lo) / span)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(pnls))
    zero_y = y(0.0) if lo <= 0 <= hi else None
    end_class = "chart-pos" if pnls[-1] >= 0 else "chart-neg"

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="pnl-chart" '
        f'preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">',
        f'<rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" class="chart-frame"/>',
    ]
    if zero_y is not None:
        parts.append(
            f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width - pad}" y2="{zero_y:.1f}" '
            f'class="chart-zero"/>'
        )
    parts.append(f'<polyline points="{points}" class="chart-line {end_class}"/>')
    parts.append(
        f'<text x="{pad}" y="{pad - 8}" class="chart-label">${hi:,.0f}</text>'
    )
    parts.append(
        f'<text x="{pad}" y="{height - pad + 18}" class="chart-label">${lo:,.0f}</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


# Convenience for `flask run` / `python -m webapp.app`
app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=True)
