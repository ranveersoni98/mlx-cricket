"""Date-ordered Elo + rolling form signals. Everything here is causal:
every feature for row i uses only rows before i.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ..config import FORM_WINDOW, H2H_WINDOW


def add_chrono_features(
    df: pd.DataFrame, k: float = 20.0, base: float = 1500.0, home_adv: float = 15.0
) -> pd.DataFrame:
    """Elo + form/H2H/venue/toss-venue/star/experience, in date order.

    Needs: team1, team2, venue_city, venue_country, toss_side, label,
    date (YYYY-MM-DD, "" falls back to year ordering).
    Optional: player_of_match (star proxy); absent -> 0.
    """
    df = df.reset_index(drop=True).copy()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * len(df))
    fallback = pd.to_datetime(df["year"].astype(str) + "-06-15", errors="coerce")
    df["_ord_date"] = dates.fillna(fallback)
    order = np.argsort(df["_ord_date"].values, kind="stable")

    home_team = {
        "India": ["India", "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
                  "Kolkata Knight Riders", "Delhi Daredevils", "Rajasthan Royals",
                  "Sunrisers Hyderabad", "Kings XI Punjab", "Deccan Chargers",
                  "Pune Warriors", "Kochi Tuskers Kerala", "Gujarat Lions",
                  "Rising Pune Supergiant"],
        "Australia": ["Australia"], "England": ["England"], "Pakistan": ["Pakistan"],
        "South Africa": ["South Africa"], "New Zealand": ["New Zealand"],
        "Sri Lanka": ["Sri Lanka"], "West Indies": ["West Indies"],
        "Bangladesh": ["Bangladesh"], "Zimbabwe": ["Zimbabwe"],
        "Afghanistan": ["Afghanistan"], "Ireland": ["Ireland"],
    }
    country_home: dict[str, str] = {}
    for c, teams in home_team.items():
        for t in teams:
            country_home[t] = c

    ratings: dict[str, float] = defaultdict(lambda: base)
    recent: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h: dict[tuple[str, str], deque[str]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    city_wins: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # (team, city) -> [wins, games]
    city_toss: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # city -> [toss-winner wins, games]
    city_draw: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # city -> [draws, TEST games]
    pom_last20: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
    games: dict[str, int] = defaultdict(int)

    n = len(df)
    e1 = np.zeros(n); e2 = np.zeros(n)
    form = np.zeros(n); hh = np.full(n, 0.5); vedge = np.zeros(n)
    tven = np.full(n, 0.5); star = np.zeros(n); expd = np.zeros(n)
    drw = np.full(n, 0.3)

    has_pom = "player_of_match" in df.columns
    for pos, idx in enumerate(order):
        r = df.iloc[int(idx)]
        t1, t2 = str(r["team1"]), str(r["team2"])
        city = str(r.get("venue_city", "Unknown"))
        r1, r2 = ratings[t1], ratings[t2]
        b1 = home_adv if country_home.get(t1) == r["venue_country"] else 0.0
        b2 = home_adv if country_home.get(t2) == r["venue_country"] else 0.0
        e1[int(idx)], e2[int(idx)] = r1 + b1, r2 + b2

        # form: mean of last-5 results (1 = won), prior 0.5 when empty
        f1 = float(np.mean(list(recent[t1]))) if recent[t1] else 0.5
        f2 = float(np.mean(list(recent[t2]))) if recent[t2] else 0.5
        form[int(idx)] = f1 - f2
        # h2h from team1 perspective
        key: tuple[str, str] = (t1, t2) if t1 <= t2 else (t2, t1)
        if h2h[key]:
            wins = len([wname for wname in h2h[key] if wname == t1])
            hh[int(idx)] = wins / len(h2h[key])
        # venue edge: city win% diff with regression to 0.5 (min 3 games)
        def city_rate(team: str) -> float:
            w, g = city_wins[(team, city)]
            return (w + 1.5) / (g + 3.0)
        vedge[int(idx)] = city_rate(t1) - city_rate(t2)
        # toss-venue: P(toss winner wins at this city), regressed
        tw, tg = city_toss[city]
        tven[int(idx)] = (tw + 1.0) / (tg + 2.0)
        # draw-venue: P(draw at this city for multi-day games), regressed
        dw, dg = city_draw[city]
        drw[int(idx)] = (dw + 1.5) / (dg + 5.0)
        # star proxy: POM share last 20 per team
        s1 = float(np.mean(list(pom_last20[t1]))) if pom_last20[t1] else 0.0
        s2 = float(np.mean(list(pom_last20[t2]))) if pom_last20[t2] else 0.0
        star[int(idx)] = s1 - s2
        expd[int(idx)] = math.log1p(games[t1]) - math.log1p(games[t2])

        # --- update with outcome (label 0/1; label 2 draws handled by caller) ---
        lab = r["label"]
        if lab in (0, 1):
            s_1 = 1.0 if lab == 0 else 0.0
            exp1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
            ratings[t1] = r1 + k * (s_1 - exp1)
            ratings[t2] = r2 + k * ((1 - s_1) - (1 - exp1))
            recent[t1].append(1.0 if lab == 0 else 0.0)
            recent[t2].append(1.0 if lab == 1 else 0.0)
            winner = t1 if lab == 0 else t2
            h2h[key].append(winner)
            city_wins[(t1, city)][1] += 1
            city_wins[(t2, city)][1] += 1
            city_wins[(winner, city)][0] += 1
            if str(r.get("toss_side", "none")) in ("team1", "team2"):
                tg1 = city_toss[city]
                tg1[1] += 1
                toss_winner = t1 if r["toss_side"] == "team1" else t2
                if toss_winner == winner:
                    tg1[0] += 1
            games[t1] += 1
            games[t2] += 1
            pom = str(r.get("player_of_match", "") or "") if has_pom else ""
            # attribute POM to winner team (best-effort; names not mapped to teams)
            pom_last20[t1].append(1.0 if pom and lab == 0 else 0.0)
            pom_last20[t2].append(1.0 if pom and lab == 1 else 0.0)
        else:
            # draw: no Elo move, but count experience + venue draw rate
            games[t1] += 1
            games[t2] += 1
            if r.get("format") == "TEST":
                cd = city_draw[city]
                cd[1] += 1
                cd[0] += 1
        if lab in (0, 1) and r.get("format") == "TEST":
            city_draw[city][1] += 1

    df["elo1"] = e1
    df["elo2"] = e2
    df["elo_diff"] = (e1 - e2) / 400.0
    df["abs_elo_diff"] = df["elo_diff"].abs()  # mismatch size: close teams draw more
    df["elo_win_prob"] = 1.0 / (1.0 + 10 ** (-df["elo_diff"].values))
    df["form_diff"] = form
    df["h2h"] = hh
    df["venue_edge"] = vedge
    df["toss_venue"] = tven
    df["star_diff"] = star
    df["exp_diff"] = expd
    df["draw_venue"] = drw
    return df.drop(columns=["_ord_date"])


def compute_elo(
    df: pd.DataFrame, k: float = 20.0, base: float = 1500.0, home_adv: float = 15.0
) -> pd.DataFrame:
    """Back-compat wrapper: full chrono features (Elo + form + ...)."""
    return add_chrono_features(df, k=k, base=base, home_adv=home_adv)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid-search temperature T minimizing val NLL (fixes overconfidence)."""
    best_t, best_nll = 1.0, float("inf")
    for t in list(np.arange(0.5, 3.01, 0.05)):
        scaled = logits / t
        m = scaled.max(axis=1, keepdims=True)
        logp = scaled - m - np.log(np.exp(scaled - m).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_nll, best_t = float(nll), float(t)
    return best_t
