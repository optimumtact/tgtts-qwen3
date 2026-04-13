#!/usr/bin/env python3
import re
import json
import argparse
import statistics
from collections import defaultdict

# Regex patterns for parsing the logs
VOICE_PATTERN = re.compile(r"Voice:\s*(.*?)\s*\|")
TTS_PATTERN = re.compile(r"TTS\s*Time:\s*([0-9.]+)s")

def parse_log_file(filepath):
    voice_data = defaultdict(list)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except FileNotFoundError:
        return {}

    entries = re.split(r"(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})", content)
    for entry in entries:
        v_m, t_m = VOICE_PATTERN.search(entry), TTS_PATTERN.search(entry)
        if v_m and t_m:
            voice_data[v_m.group(1).strip()].append(float(t_m.group(1)))
    return voice_data

def get_stats(voice_data):
    """Process data with Slowness/Fastness logic based on P95 thresholds."""
    all_times = [t for times in voice_data.values() for t in times]
    if not all_times: return None, []
    
    all_times.sort()
    # Calculate global P95 for the corpus as a benchmark
    global_p95 = all_times[max(0, int(len(all_times) * 0.95) - 1)]
    # Use global quantiles of P95s or just standard deviation for categorization
    # Here we define thresholds based on the global distribution's quartiles for simplicity
    global_q1, _, global_q3 = statistics.quantiles(all_times, n=4)
    
    def process_entry(name, times, is_global=False):
        times.sort()
        p95_val = times[max(0, int(len(times) * 0.95) - 1)] if times else 0
        v_median = statistics.median(times)
        
        return {
            "voice": name,
            "values": times,
            "median": v_median,
            "p95": p95_val,
            "min": min(times),
            "max": max(times),
            "count": len(times),
            # CATEGORIZATION LOGIC: Based on P95 performance
            "is_slow": p95_val >= global_q3 if not is_global else False,
            "is_fast": p95_val <= global_q1 if not is_global else False,
            "is_global": is_global
        }

    global_stats = process_entry("GLOBAL CORPUS", all_times, True)
    stats_list = [process_entry(v, t) for v, t in voice_data.items()]
    # Sort sidebar by P95 descending so worst offenders are at the top
    stats_list.sort(key=lambda x: x["p95"], reverse=True)
    return global_stats, stats_list

def generate_html(global_stats, voice_data, output_path):
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TTS Latency Dashboard (P95 Optimized)</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background: #fdfdfd; color: #333; }}
        header {{ background: #232f3e; color: white; padding: 15px 25px; display: flex; justify-content: space-between; align-items: center; }}
        .main-container {{ display: flex; flex: 1; overflow: hidden; }}
        #sidebar {{ width: 380px; background: white; border-right: 1px solid #ddd; display: flex; flex-direction: column; }}
        .sidebar-header {{ padding: 15px; border-bottom: 1px solid #eee; }}
        #search-bar {{ width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }}
        #voice-list {{ flex: 1; overflow-y: auto; padding: 10px; }}
        #chart-container {{ flex: 1; padding: 20px; overflow-y: auto; background: #fff; }}
        
        .info-section {{ margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 5px solid #232f3e; max-width: 900px; }}
        .key-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .key-item {{ display: flex; align-items: center; font-size: 13px; }}
        .key-box {{ width: 14px; height: 14px; margin-right: 8px; border-radius: 2px; border: 1px solid #333; }}
        .symbol {{ font-size: 16px; margin-right: 8px; width: 20px; text-align: center; font-weight: bold; }}
        
        .voice-item {{ display: flex; align-items: center; padding: 8px; font-size: 13px; border-bottom: 1px solid #f9f9f9; }}
        .voice-item:hover {{ background: #f1f5f9; }}
        .v-meta {{ font-size: 11px; color: #666; margin-left: 10px; }}
        .badge {{ margin-left: auto; font-size: 10px; padding: 2px 6px; border-radius: 10px; font-weight: bold; }}
        .slow-badge {{ background: #fee2e2; color: #b91c1c; }}
        .fast-badge {{ background: #dcfce7; color: #15803d; }}
        
        .violin {{ stroke: #333; stroke-width: 0.5; opacity: 0.6; cursor: pointer; }}
        .range-line {{ stroke: #333; stroke-width: 1.5; opacity: 0.3; }}
        .range-cap {{ stroke: #333; stroke-width: 2; }}
        .median-dot {{ fill: white; stroke: #333; stroke-width: 1.5; }}
        .p95-dot {{ fill: #ff4d4d; stroke: #b30000; stroke-width: 1; }}
        .tooltip {{ position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 12px; border-radius: 8px; pointer-events: none; font-size: 12px; line-height: 1.5; z-index: 100; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1 style="margin:0; font-size: 20px;">TTS Latency Analyzer</h1>
            <small style="opacity:0.8;">Categorization based on 95th Percentile (P95) stability</small>
        </div>
    </header>
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
                    <div class="key-item"><div class="key-box" style="background: #e74c3c;"></div> High P95 (Unstable/Slow)</div>
                    <div class="key-item"><div class="key-box" style="background: #2ecc71;"></div> Low P95 (Fast/Consistent)</div>
                    <div class="key-item"><div class="key-box" style="background: #3498db;"></div> Standard Range</div>
                    <div class="key-item"><span class="symbol" style="color:#000;">○</span> <b>Median:</b> Typical 50% Speed</div>
                    <div class="key-item"><span class="symbol" style="color:#ff4d4d;">●</span> <b>P95:</b> 95% of calls are faster than this</div>
                    <div class="key-item"><span class="symbol">╤</span> <b>Max/Min:</b> Peak latency range</div>
                </div>
                <p class="explainer-text" style="font-size:13px; color:#666;">
                    <b>Logic Change:</b> Voices are now flagged as "Slow" or "Fast" based on their <b>P95</b> value rather than the median. This highlights voices that might have a fast average but suffer from severe, unpredictable spikes.
                </p>
            </div>
        </div>
    </div>

    <script>
        const globalStats = {json.dumps(global_stats)};
        const voices = {json.dumps(voice_data)};
        let activeVoices = [];
        const tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);

        function filterVoices() {{
            const term = document.getElementById('search-bar').value.toLowerCase();
            d3.selectAll(".voice-item").style("display", function() {{
                return d3.select(this).select(".v-name").text().toLowerCase().includes(term) ? "flex" : "none";
            }});
        }}

        function toggleAll(state) {{
            d3.selectAll(".voice-checkbox").property("checked", state);
            activeVoices = state ? voices.map(v => v.voice) : [];
            renderChart();
        }}

        function kernelDensityEstimator(kernel, X) {{
            return function(V) {{
                return X.map(x => [x, d3.mean(V, v => kernel(x - v))]);
            }};
        }}
        function kernelEpanechnikov(k) {{
            return v => Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
        }}

        function renderChart() {{
            d3.select("#chart").selectAll("*").remove();
            const displayData = [globalStats, ...voices.filter(v => activeVoices.includes(v.voice))];
            
            const margin = {{top: 40, right: 30, bottom: 120, left: 60}},
                  width = Math.max(900, displayData.length * 100) - margin.left - margin.right,
                  height = 500 - margin.top - margin.bottom;

            const svg = d3.select("#chart").append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            const x = d3.scalePoint().range([0, width]).domain(displayData.map(d => d.voice)).padding(0.5);
            const yMax = d3.max(voices, d => d.max);
            const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([height, 0]);

            svg.append("g").attr("transform", `translate(0,${{height}})`).call(d3.axisBottom(x))
                .selectAll("text").attr("transform", "rotate(-45)").style("text-anchor", "end");
            svg.append("g").call(d3.axisLeft(y).tickFormat(d => d + "s"));

            const kde = kernelDensityEstimator(kernelEpanechnikov(.1), y.ticks(60));
            const violinWidth = 50;

            displayData.forEach(d => {{
                const density = kde(d.values);
                const xPos = x(d.voice);
                const maxNum = d3.max(density, v => v[1]);
                const xNum = d3.scaleLinear().range([0, violinWidth / 2]).domain([0, maxNum]);
                const color = d.is_global ? "#9b59b6" : (d.is_slow ? "#e74c3c" : (d.is_fast ? "#2ecc71" : "#3498db"));

                svg.append("line").attr("class", "range-line").attr("x1", xPos).attr("x2", xPos).attr("y1", y(d.min)).attr("y2", y(d.max));
                const capW = 6;
                svg.append("line").attr("class", "range-cap").attr("x1", xPos-capW).attr("x2", xPos+capW).attr("y1", y(d.min)).attr("y2", y(d.min));
                svg.append("line").attr("class", "range-cap").attr("x1", xPos-capW).attr("x2", xPos+capW).attr("y1", y(d.max)).attr("y2", y(d.max));

                svg.append("path")
                    .datum(density)
                    .attr("class", "violin")
                    .style("fill", color)
                    .attr("d", d3.area().x0(v => -xNum(v[1])).x1(v => xNum(v[1])).y(v => y(v[0])).curve(d3.curveCatmullRom))
                    .attr("transform", `translate(${{xPos}},0)`)
                    .on("mouseover", (event) => {{
                        tooltip.transition().duration(200).style("opacity", .9);
                        tooltip.html(`
                            <b style="font-size:14px; color:#3498db">${{d.voice}}</b><hr style="border:0; border-top:1px solid #444;">
                            Min: ${{d.min.toFixed(3)}}s<br>
                            Median: ${{d.median.toFixed(3)}}s<br>
                            <span style="color:#ff4d4d">P95: ${{d.p95.toFixed(3)}}s</span><br>
                            Max: ${{d.max.toFixed(3)}}s<br>
                            Samples: ${{d.count}}
                        `)
                        .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 28) + "px");
                    }})
                    .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));

                svg.append("circle").attr("class", "median-dot").attr("cx", xPos).attr("cy", y(d.median)).attr("r", 4);
                svg.append("circle").attr("class", "p95-dot").attr("cx", xPos).attr("cy", y(d.p95)).attr("r", 3.5);
            }});
        }}

        const list = d3.select("#voice-list");
        voices.forEach(d => {{
            const item = list.append("div").attr("class", "voice-item");
            item.append("input").attr("type", "checkbox").attr("class", "voice-checkbox")
                .on("change", function() {{
                    if(this.checked) activeVoices.push(d.voice);
                    else activeVoices = activeVoices.filter(v => v !== d.voice);
                    renderChart();
                }});
            item.append("span").attr("class", "v-name").text(d.voice);
            item.append("span").attr("class", "v-meta").text(`p95: ${{d.p95.toFixed(2)}}s`);
            
            if(d.is_slow) item.append("span").attr("class", "badge slow-badge").text("SLOW");
            if(d.is_fast) item.append("span").attr("class", "badge fast-badge").text("FAST");
        }});
        renderChart();
    </script>
</body>
</html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\nDashboard Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--output", default="tts_p95_dashboard.html")
    args = parser.parse_args()
    data = parse_log_file(args.logfile)
    if data:
        g, v = get_stats(data)
        generate_html(g, v, args.output)

if __name__ == "__main__":
    main()
