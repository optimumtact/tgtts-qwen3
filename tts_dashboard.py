import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

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
    # 1. Load the data
    df = pd.read_sql_query(query, conn)
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


@app.route("/api/stats")
def get_stats():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    voices = request.args.get("voices")

    query = "SELECT * FROM tts_logs"
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

    conn = get_db_connection()
    logs = conn.execute(query, params).fetchall()
    conn.close()

    data = [dict(log) for log in logs]
    return jsonify(data)


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
        <div style="display:flex; gap:5px; align-items:center; margin-right: 15px; border-right: 1px solid #ccc; padding-right: 15px;">
            <label>Quick Range:</label>
            <button onclick="setRange(30)">30m</button>
            <button onclick="setRange(360)">6h</button>
            <button onclick="setRange(1440)">24h</button>
        </div>
        <label>Start Date:</label>
        <input type="datetime-local" id="start-date">
        <label>End Date:</label>
        <input type="datetime-local" id="end-date">
        <button onclick="fetchData()">Filter</button>
        <button onclick="resetFilters()">Reset</button>
        <div style="margin-left: auto;">
            <label><input type="checkbox" id="auto-refresh" checked> Auto-refresh (30s)</label>
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
            <h3>Requests per Minute</h3>
            <div id="requests-chart"></div>
        </div>
        <div class="traffic-chart-box">
            <h3>Median Total Time (per minute)</h3>
            <div id="latency-over-time-chart"></div>
        </div>
    </div>
    <script src="/static/visualisations.js"></script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007)
