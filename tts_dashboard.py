import os
import sqlite3
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
DB_PATH = os.getenv("DB_PATH", "/workspace/tts_stats.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    return conn


@app.route("/api/stats")
def get_stats():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

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
    <style>
        body { font-family: -apple-system, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background: #fdfdfd; color: #333; }
        header { background: #232f3e; color: white; padding: 15px 25px; display: flex; justify-content: space-between; align-items: center; }
        .main-container { display: flex; flex: 1; overflow: hidden; }
        #sidebar { width: 380px; background: white; border-right: 1px solid #ddd; display: flex; flex-direction: column; }
        .sidebar-header { padding: 15px; border-bottom: 1px solid #eee; }
        #search-bar { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        #voice-list { flex: 1; overflow-y: auto; padding: 10px; }
        #chart-container { flex: 1; padding: 20px; overflow-y: auto; background: #fff; }
        
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
        <div id="last-update" style="font-size: 12px; opacity: 0.8;"></div>
    </header>
    <div class="controls">
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
    <div class="main-container">
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

    <script>
        let rawData = [];
        let voiceStats = [];
        let globalStats = {};
        let activeVoices = [];
        const tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);

        async function fetchData() {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            
            let url = '/api/stats';
            const params = new URLSearchParams();
            if (startDate) params.append('start_date', startDate.replace('T', ' '));
            if (endDate) params.append('end_date', endDate.replace('T', ' '));
            if (params.toString()) url += '?' + params.toString();

            try {
                const response = await fetch(url);
                rawData = await response.json();
                processData();
                renderVoiceList();
                renderChart();
                document.getElementById('last-update').innerText = 'Last updated: ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.error("Failed to fetch data", e);
            }
        }

        function processData() {
            const voiceDataMap = new Map();
            const allTimes = [];

            rawData.forEach(row => {
                const voice = row.voice_used || 'unknown';
                const time = row.tts_time;
                if (!voiceDataMap.has(voice)) voiceDataMap.set(voice, []);
                voiceDataMap.get(voice).push(time);
                allTimes.push(time);
            });

            if (allTimes.length === 0) {
                voiceStats = [];
                globalStats = { voice: "GLOBAL CORPUS", values: [], median: 0, p95: 0, min: 0, max: 0, count: 0, is_global: true };
                return;
            }

            allTimes.sort((a, b) => a - b);
            const globalQ1 = d3.quantile(allTimes, 0.25);
            const globalQ3 = d3.quantile(allTimes, 0.75);

            function getStats(name, times, isGlobal = false) {
                times.sort((a, b) => a - b);
                const p95 = d3.quantile(times, 0.95);
                return {
                    voice: name,
                    values: times,
                    median: d3.median(times),
                    p95: p95,
                    min: d3.min(times),
                    max: d3.max(times),
                    count: times.length,
                    is_slow: !isGlobal && p95 >= globalQ3,
                    is_fast: !isGlobal && p95 <= globalQ1,
                    is_global: isGlobal,
                    is_insignificant: !isGlobal && times.length < 20
                };
            }

            globalStats = getStats("GLOBAL CORPUS", allTimes, true);
            voiceStats = Array.from(voiceDataMap.entries()).map(([name, times]) => getStats(name, times));
            voiceStats.sort((a, b) => b.p95 - a.p95);
        }

        function renderVoiceList() {
            const list = d3.select("#voice-list");
            const term = document.getElementById('search-bar').value.toLowerCase();
            list.selectAll("*").remove();

            voiceStats.forEach(d => {
                const isChecked = activeVoices.includes(d.voice);
                const item = list.append("div").attr("class", "voice-item")
                    .style("display", d.voice.toLowerCase().includes(term) ? "flex" : "none")
                    .style("background", d.is_insignificant ? "#fff7ed" : "transparent");
                
                item.append("input").attr("type", "checkbox").attr("class", "voice-checkbox")
                    .property("checked", isChecked)
                    .on("change", function() {
                        if(this.checked) {
                            if (!activeVoices.includes(d.voice)) activeVoices.push(d.voice);
                        } else {
                            activeVoices = activeVoices.filter(v => v !== d.voice);
                        }
                        renderChart();
                    });
                
                item.append("span").attr("class", "v-name").text(`${d.voice} (${d.count})`);
                item.append("span").attr("class", "v-meta").text(`p95: ${d.p95.toFixed(2)}s`);
                
                if(d.is_insignificant) item.append("span").attr("class", "badge orange-badge").text("LOW SAMPLES");
                if(d.is_slow) item.append("span").attr("class", "badge slow-badge").text("SLOW");
                if(d.is_fast) item.append("span").attr("class", "badge fast-badge").text("FAST");
            });
        }

        function filterVoices() {
            const term = document.getElementById('search-bar').value.toLowerCase();
            d3.selectAll(".voice-item").each(function() {
                const nameText = d3.select(this).select(".v-name").text().toLowerCase();
                const nameMatch = nameText.split(' (')[0].includes(term); // Only match voice name part
                d3.select(this).style("display", nameMatch ? "flex" : "none");
            });
        }

        function toggleAll(state) {
            d3.selectAll(".voice-checkbox").property("checked", state);
            if (state) {
                activeVoices = voiceStats.map(v => v.voice);
            } else {
                activeVoices = [];
            }
            renderChart();
        }

        function resetFilters() {
            document.getElementById('start-date').value = '';
            document.getElementById('end-date').value = '';
            fetchData();
        }

        function kernelDensityEstimator(kernel, X) {
            return function(V) {
                return X.map(x => [x, d3.mean(V, v => kernel(x - v))]);
            };
        }
        function kernelEpanechnikov(k) {
            return v => Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
        }

        function renderChart() {
            d3.select("#chart").selectAll("*").remove();
            if (voiceStats.length === 0 && !globalStats.values.length) {
                d3.select("#chart").append("div").text("No data available for the selected range.").style("padding", "20px");
                return;
            }

            const displayData = [globalStats, ...voiceStats.filter(v => activeVoices.includes(v.voice))];
            
            const margin = {top: 40, right: 30, bottom: 120, left: 60},
                  width = Math.max(900, displayData.length * 100) - margin.left - margin.right,
                  height = 500 - margin.top - margin.bottom;

            const svg = d3.select("#chart").append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

            const x = d3.scalePoint().range([0, width]).domain(displayData.map(d => d.voice)).padding(0.5);
            const yMax = d3.max(voiceStats.length ? voiceStats : [globalStats], d => d.max) || 1;
            const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([height, 0]);

            svg.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x))
                .selectAll("text").attr("transform", "rotate(-45)").style("text-anchor", "end");
            svg.append("g").call(d3.axisLeft(y).tickFormat(d => d + "s"));

            const kde = kernelDensityEstimator(kernelEpanechnikov(.1), y.ticks(60));
            const violinWidth = 50;

            displayData.forEach(d => {
                if (d.count === 0) return;
                const density = kde(d.values);
                const xPos = x(d.voice);
                const maxNum = d3.max(density, v => v[1]);
                const xNum = d3.scaleLinear().range([0, violinWidth / 2]).domain([0, maxNum]);
                const color = d.is_global ? "#9b59b6" : (d.is_insignificant ? "#ffa500" : (d.is_slow ? "#e74c3c" : (d.is_fast ? "#2ecc71" : "#3498db")));

                svg.append("line").attr("class", "range-line").attr("x1", xPos).attr("x2", xPos).attr("y1", y(d.min)).attr("y2", y(d.max));
                const capW = 6;
                svg.append("line").attr("class", "range-cap").attr("x1", xPos-capW).attr("x2", xPos+capW).attr("y1", y(d.min)).attr("y2", y(d.min));
                svg.append("line").attr("class", "range-cap").attr("x1", xPos-capW).attr("x2", xPos+capW).attr("y1", y(d.max)).attr("y2", y(d.max));

                svg.append("path")
                    .datum(density)
                    .attr("class", "violin")
                    .style("fill", color)
                    .attr("d", d3.area().x0(v => -xNum(v[1])).x1(v => xNum(v[1])).y(v => y(v[0])).curve(d3.curveCatmullRom))
                    .attr("transform", `translate(${xPos},0)`)
                    .on("mouseover", (event) => {
                        tooltip.transition().duration(200).style("opacity", .9);
                        tooltip.html(`
                            <b style="font-size:14px; color:#3498db">${d.voice}</b><hr style="border:0; border-top:1px solid #444;">
                            Min: ${d.min.toFixed(3)}s<br>
                            Median: ${d.median.toFixed(3)}s<br>
                            <span style="color:#ff4d4d">P95: ${d.p95.toFixed(3)}s</span><br>
                            Max: ${d.max.toFixed(3)}s<br>
                            Samples: ${d.count}
                        `)
                        .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 28) + "px");
                    })
                    .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));

                svg.append("circle").attr("class", "median-dot").attr("cx", xPos).attr("cy", y(d.median)).attr("r", 4);
                svg.append("circle").attr("class", "p95-dot").attr("cx", xPos).attr("cy", y(d.p95)).attr("r", 3.5);
            });
        }

        // Initial fetch
        fetchData();

        // Auto refresh
        setInterval(() => {
            if (document.getElementById('auto-refresh').checked) {
                fetchData();
            }
        }, 30000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007)
