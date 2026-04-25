import json
import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from scipy.stats import gaussian_kde

app = Flask(__name__)
DB_PATH = os.getenv("DB_PATH", "/workspace/tts_stats.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    return conn


@app.route("/api/latency")
def get_latency():
    conn = get_db_connection()
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    period = request.args.get("period", 60)

    query = "SELECT timestamp,total_time FROM tts_logs"
    params = []

    if start_date or end_date:
        query += " WHERE "
        if start_date:
            query += "timestamp >= ?"
            params.append(start_date)
        if end_date:
            if start_date:
                query += " AND "
            query += "timestamp <= ?"
            params.append(end_date)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # 2. Convert timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # 3. Resample by your custom period (e.g., 'D' for Day, 'H' for Hour, 'W' for Week)
    # This calculates Count, Mean, Median, and P95 all at once
    stats = (
        df["total_time"]
        .resample(f"{period}s")
        .agg(
            {
                "count": "count",
                "mean": "mean",
                "median": "median",
                "max": "max",
                "min": "min",
                "p95": lambda x: x.quantile(0.95),
            }
        )
        .fillna(0)
    )
    data = stats.reset_index().to_dict(
        orient="records"
    )  # Convert DataFrame to a list of dictionaries
    return jsonify(data)


def tts_stats(voice, total_time):
    stats = {
        "min": float(total_time.min()),
        "max": float(total_time.max()),
        "median": float(total_time.median()),
        "p95": float(total_time.quantile(0.95)),
        "q1": float(total_time.quantile(0.25)),
        "q3": float(total_time.quantile(0.75)),
        "count": int(total_time.count()),
    }

    # Kernel Density Estimate (100 points)
    if len(total_time) > 1 and total_time.std() > 0:
        try:
            kde = gaussian_kde(total_time)
            x_ind = np.linspace(total_time.min(), total_time.max(), 100)
            y_val = kde.evaluate(x_ind)
            stats["kde"] = {"x": x_ind.tolist(), "y": y_val.tolist()}
        except Exception:
            stats["kde"] = None
    else:
        stats["kde"] = None

    return {"voice": voice, "data": stats}


@app.route("/api/bands")
def get_bands():
    return jsonify([1, 2, 3, 5, "all"])


@app.route("/api/voices")
def get_voices():
    band = request.args.get("band")
    if band:
        conn = get_db_connection()
        # Fetch voices for this band from cache
        query = "SELECT voice as voice_used, count, p95 FROM cachedstats WHERE timeband = ?"
        df = pd.read_sql_query(query, conn, params=[band])
        
        # We also need global stats for this band to calculate is_slow/is_fast
        global_query = "SELECT q1, q3 FROM cachedstats WHERE voice = 'global' AND timeband = ?"
        global_res = conn.execute(global_query, [band]).fetchone()
        conn.close()
        
        global_stats = {"q1": 0, "q3": 0}
        if global_res:
            global_stats = {"q1": global_res["q1"], "q3": global_res["q3"]}
            
        return jsonify({"global": global_stats, "voices": df.to_dict(orient="records")})

    params = []
    query = ""
    start_date = request.args.get("start_date")
    # ... (rest of the existing logic for custom range)
    end_date = request.args.get("end_date")
    if start_date or end_date:
        query += " WHERE "
        if start_date:
            query += "timestamp >= ?"
            params.append(start_date)
        if end_date:
            if start_date:
                query += " AND "
            query += "timestamp <= ?"
            params.append(end_date)
    final_query = f"SELECT * FROM tts_logs {query}"
    conn = get_db_connection()
    df = pd.read_sql_query(final_query, conn, params=params)
    conn.close()
    if df.empty:
        return jsonify({"global": {"count": 0, "q3": 0, "q1": 0}, "voices": []})
    # Per-voice aggregation
    voice_stats = (
        df.groupby("voice_used")["total_time"]
        .agg(count="count", p95=lambda x: x.quantile(0.95))
        .reset_index()
    )

    # Global stats
    global_stats = {
        "count": len(df),
        "q3": df["total_time"].quantile(0.75),
        "q1": df["total_time"].quantile(0.25),
    }
    result = {"global": global_stats, "voices": voice_stats.to_dict(orient="records")}
    return jsonify(result)


@app.route("/api/ttsstats")
def get_tts_stats():
    band = request.args.get("band")
    voices = request.args.getlist("voices")
    if voices and len(voices) == 1 and "," in voices[0]:
        voices = voices[0].split(",")

    if band:
        conn = get_db_connection()
        query = "SELECT * FROM cachedstats WHERE timeband = ? AND (voice = 'global'"
        params = [band]
        if voices:
            query += " OR voice IN ({})".format(",".join(["?"] * len(voices)))
            params.extend(voices)
        query += ")"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        results = []
        for _, row in df.iterrows():
            kde = None
            if row["kde"]:
                try: 
                    kde_list = json.loads(row["kde"])
                    # Convert back to {"x": [...], "y": [...]}
                    x_vals, y_vals = zip(*kde_list)
                    kde = {"x": list(x_vals), "y": list(y_vals)}
                except Exception as e:
                    print(f"Could not load band {band} for voice {row['voice']}")
                    kde = []
            
            results.append({
                "voice": row["voice"],
                "data": {
                    "min": row["min"],
                    "max": row["max"],
                    "median": row["median"],
                    "p95": row["p95"],
                    "q1": row["q1"],
                    "q3": row["q3"],
                    "count": row["count"],
                    "kde": kde
                }
            })
        return jsonify(results)

    start_date = request.args.get("start_date")
    # ... (rest of existing logic)


@app.route("/api/regenerate")
def regenerate_stats():
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cachedstats (
                voice TEXT,
                timeband TEXT,
                min REAL,
                max REAL,
                median REAL,
                p95 REAL,
                q1 REAL,
                q3 REAL,
                count INTEGER,
                kde TEXT,
                PRIMARY KEY (voice, timeband)
            )
        """
        )

        timebands = [1, 2, 3, 5, "all"]
        now = datetime.now()

        for band in timebands:
            query = "SELECT voice_used, total_time FROM tts_logs"
            params = []
            if band != "all":
                start_date = (now - timedelta(days=int(band))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                query += " WHERE timestamp >= ?"
                params.append(start_date)

            df = pd.read_sql_query(query, conn, params=params)

            if df.empty:
                continue

            # Process Global
            global_res = tts_stats("global", df["total_time"])
            _save_to_cache(conn, str(band), global_res)

            # Process Voices
            for voice in df["voice_used"].unique():
                voice_df = df[df["voice_used"] == voice]
                if not voice_df.empty:
                    voice_res = tts_stats(voice, voice_df["total_time"])
                    _save_to_cache(conn, str(band), voice_res)

        conn.commit()
        return jsonify({"status": "success", "message": "Stats regenerated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


def _save_to_cache(conn, band, stat_result):
    voice = stat_result["voice"]
    data = stat_result["data"]

    # "the kde stat should be stored as a json list"
    kde_json = None
    if data["kde"]:
        # Converting {"x": [...], "y": [...]} to [[x1, y1], [x2, y2], ...]
        kde_list = list(zip(data["kde"]["x"], data["kde"]["y"]))
        kde_json = json.dumps(kde_list)

    conn.execute(
        """
        INSERT OR REPLACE INTO cachedstats 
        (voice, timeband, min, max, median, p95, q1, q3, count, kde)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            voice,
            band,
            data["min"],
            data["max"],
            data["median"],
            data["p95"],
            data["q1"],
            data["q3"],
            data["count"],
            kde_json,
        ),
    )


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TTS Latency Dashboard (Real-time)</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
    <style>
        body { font-family: -apple-system, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background: #fdfdfd; color: #333; }
        header { background: #232f3e; color: white; padding: 15px 25px; display: flex; justify-content: space-between; align-items: center; }
        .nav-links { display: flex; gap: 20px; }
        .nav-links a { color: #fff; text-decoration: none; font-weight: 500; opacity: 0.8; cursor: pointer; }
        .nav-links a.active { opacity: 1; border-bottom: 2px solid #fff; }
        .nav-links a:hover { opacity: 1; }

        .main-container { display: flex; flex: 1; overflow: hidden; }
        #sidebar { width: 380px; background: white; border-right: 1px solid #ddd; display: flex; flex-direction: column; }
        .sidebar-header { padding: 15px; border-bottom: 1px solid #eee; }
        #search-bar { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        #voice-list { flex: 1; overflow-y: auto; padding: 10px; }
        #chart-container { flex: 1; padding: 20px; overflow-y: auto; background: #fff; }
        
        #traffic-page { display: none; flex: 1; padding: 20px; overflow-y: auto; background: #fff; flex-direction: column; }
        .traffic-chart-box { background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .traffic-chart-box h3 { margin-top: 0; color: #232f3e; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 10px; }

        .info-section { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 5px solid #232f3e; max-width: 900px; }
        .key-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .key-item { display: flex; align-items: center; font-size: 13px; }
        .key-box { width: 14px; height: 14px; margin-right: 8px; border-radius: 2px; border: 1px solid #333; }
        .symbol { font-size: 16px; margin-right: 8px; width: 20px; text-align: center; font-weight: bold; }
        
        .voice-item { display: flex; align-items: center; padding: 8px; font-size: 13px; border-bottom: 1px solid #f9f9f9; }
        .voice-item:hover { background: #f1f5f9; }
        .v-meta { font-size: 11px; color: #666; margin-left: 10px; }
        .badge { margin-left: auto; font-size: 10px; padding: 2px 6px; border-radius: 10px; font-weight: bold; }
        .slow-badge { background: #fee2e2; color: #b91c1c; }
        .fast-badge { background: #dcfce7; color: #15803d; }
        .orange-badge { background: #ffedd5; color: #9a3412; }
        
        .violin { stroke: #333; stroke-width: 0.5; opacity: 0.6; cursor: pointer; }
        .range-line { stroke: #333; stroke-width: 1.5; opacity: 0.3; }
        .range-cap { stroke: #333; stroke-width: 2; }
        .median-dot { fill: white; stroke: #333; stroke-width: 1.5; }
        .p95-dot { fill: #ff4d4d; stroke: #b30000; stroke-width: 1; }
        
        .bar { fill: #3498db; }
        .bar:hover { fill: #2980b9; }
        .line { fill: none; stroke: #e74c3c; stroke-width: 2; }
        .dot { fill: #e74c3c; }

        .tooltip { position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 12px; border-radius: 8px; pointer-events: none; font-size: 12px; line-height: 1.5; z-index: 100; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }

        .controls { display: flex; gap: 10px; align-items: center; padding: 10px 25px; background: #eee; border-bottom: 1px solid #ddd; }
        .controls input { padding: 5px; border: 1px solid #ccc; border-radius: 4px; }

        .spinner {
            width: 12px;
            height: 12px;
            border: 2px solid #333;
            border-top: 2px solid transparent;
            border-radius: 50%;
            display: inline-block;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        #regenerate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        button.active {
            background-color: #232f3e;
            color: white;
            border-color: #232f3e;
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1 style="margin:0; font-size: 20px;">TTS Latency Analyzer (Real-time)</h1>
            <small style="opacity:0.8;">Categorization based on 95th Percentile (P95) stability</small>
        </div>
        <div class="nav-links">
            <a id="nav-latency" class="active" onclick="showPage('latency')">Latency Stats</a>
            <a id="nav-traffic" onclick="showPage('traffic')">Traffic Analysis</a>
        </div>
        <div id="last-update" style="font-size: 12px; opacity: 0.8;"></div>
    </header>
    <div class="controls">
        <div id="latency-controls" style="display:flex; gap:10px; align-items:center;">
            <label>Time Band:</label>
            <div id="band-buttons" style="display:flex; gap:5px;">
                <!-- Buttons will be injected here -->
            </div>
        </div>
        <div id="manual-range-controls" style="display:none; gap:10px; align-items:center;">
            <div style="display:flex; gap:5px; align-items:center; margin-right: 15px; border-right: 1px solid #ccc; padding-right: 15px;">
                <label>Quick Range:</label>
                <button onclick="setRange(30)">30m</button>
                <button onclick="setRange(180)">3h</button>
                <button onclick="setRange(360)">6h</button>
                <button onclick="setRange(1440)">24h</button>
                <button onclick="setRange(2880)">2d</button>
                <button onclick="setRange(7200)">5d</button>
            </div>
            <label>Start Date:</label>
            <input type="datetime-local" id="start-date">
            <label>End Date:</label>
            <input type="datetime-local" id="end-date">
            <button onclick="fetchData()">Filter</button>
            <button onclick="resetFilters()">Reset</button>
        </div>
        <div style="margin-left: auto; display: flex; align-items: center; gap: 10px;">
             <button id="regenerate-btn" onclick="regenerateStats()" style="display: flex; align-items: center; gap: 8px;">
                Regenerate Stats
                <span id="regenerate-spinner" class="spinner" style="display: none;"></span>
            </button>
        </div>
    </div>
    
    <div id="latency-page" class="main-container">
        <div id="sidebar">
            <div class="sidebar-header">
                <input type="text" id="search-bar" placeholder="Search voices..." onkeyup="filterVoices()">
                <div style="margin-top:10px; display:flex; gap:5px;">
                    <button onclick="toggleAll(true)">All</button> 
                    <button onclick="toggleAll(false)">None</button>
                </div>
            </div>
            <div id="voice-list"></div>
        </div>
        <div id="chart-container">
            <div id="chart"></div>
            
            <div class="info-section">
                <h3>Visual Key & Methodology</h3>
                <div class="key-grid">
                    <div class="key-item"><div class="key-box" style="background: #9b59b6;"></div> Global Corpus</div>
                    <div class="key-item"><div class="key-box" style="background: #ffa500;"></div> Insignificant (< 20 samples)</div>
                    <div class="key-item"><div class="key-box" style="background: #e74c3c;"></div> High P95 (Unstable/Slow)</div>
                    <div class="key-item"><div class="key-box" style="background: #2ecc71;"></div> Low P95 (Fast/Consistent)</div>
                    <div class="key-item"><div class="key-box" style="background: #3498db;"></div> Standard Range</div>
                    <div class="key-item"><span class="symbol" style="color:#000;">○</span> <b>Median:</b> Typical 50% Speed</div>
                    <div class="key-item"><span class="symbol" style="color:#ff4d4d;">●</span> <b>P95:</b> 95% of calls are faster than this</div>
                    <div class="key-item"><span class="symbol">╤</span> <b>Max/Min:</b> Peak latency range</div>
                </div>
            </div>
        </div>
    </div>

    <div id="traffic-page">
        <div class="traffic-chart-box">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; margin-bottom: 15px; padding-bottom: 10px;">
                <h3 style="margin:0;">Traffic & Latency Analysis</h3>
                <label style="font-size: 13px; font-weight: 500; cursor: pointer;">
                    <input type="checkbox" id="auto-refresh" onchange="toggleAutoRefresh()"> 
                    Live Sliding Window (Last 1h)
                </label>
            </div>
            <div style="display: flex; gap: 20px; margin-bottom: 15px; font-size: 13px; justify-content: center; flex-wrap: wrap;">
                <div style="display: flex; align-items: center;"><div style="width: 20px; height: 2px; background: #3498db; margin-right: 8px;"></div> Median</div>
                <div style="display: flex; align-items: center;"><div style="width: 20px; height: 2px; border-top: 2px dashed #2ecc71; margin-right: 8px;"></div> Mean</div>
                <div style="display: flex; align-items: center;"><div style="width: 20px; height: 12px; background: #3498db; opacity: 0.15; margin-right: 8px;"></div> Range (Min-P95)</div>
                <div style="display: flex; align-items: center;"><div style="width: 10px; height: 10px; background: #e74c3c; border-radius: 50%; margin-right: 8px;"></div> Request Count</div>
            </div>
            <div id="traffic-combined-chart"></div>
        </div>
    </div>
    <script src="/static/visualisations.js"></script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007)
