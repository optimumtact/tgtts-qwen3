const DateTime = luxon.DateTime;
let rawData = [];
let voiceStats = [];
let globalStats = {};
let activeVoices = [];
let currentPage = 'latency';
const tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);

function toISOStringLocal(date) {
    const pad = n => n.toString().padStart(2, '0');
    return date.getFullYear() + '-' + 
           pad(date.getMonth() + 1) + '-' + 
           pad(date.getDate()) + 'T' + 
           pad(date.getHours()) + ':' + 
           pad(date.getMinutes());
}

function setRange(minutes) {
    const now = DateTime.now();
    const start = now.minus({ minutes: minutes });
    document.getElementById('start-date').value = start.toFormat("yyyy-MM-dd'T'HH:mm");
    document.getElementById('end-date').value = now.toFormat("yyyy-MM-dd'T'HH:mm");
    fetchData();
}

function showPage(page) {
    currentPage = page;
    document.getElementById('latency-page').style.display = (page === 'latency') ? 'flex' : 'none';
    document.getElementById('traffic-page').style.display = (page === 'traffic') ? 'flex' : 'none';
    
    document.getElementById('nav-latency').classList.toggle('active', page === 'latency');
    document.getElementById('nav-traffic').classList.toggle('active', page === 'traffic');
    
    if (page === 'latency') renderChart();
    else renderTrafficCharts();
}

async function fetchData() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    let url = '/api/stats';
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', DateTime.fromISO(startDate).toUTC().toFormat('yyyy-MM-dd HH:mm:ss'));
    if (endDate) params.append('end_date', DateTime.fromISO(endDate).toUTC().toFormat('yyyy-MM-dd HH:mm:ss'));
    if (params.toString()) url += '?' + params.toString();

    try {
        const response = await fetch(url);
        rawData = await response.json();
        processData();
        renderVoiceList();
        
        if (currentPage === 'latency') renderChart();
        else renderTrafficCharts();
        
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

function processTrafficData() {
    const minuteData = new Map();
    rawData.forEach(row => {
        // SQLite timestamps are usually UTC
        const ts = row.timestamp.includes(' ') ? row.timestamp.replace(' ', 'T') + 'Z' : row.timestamp;
        const date = new Date(ts);
        date.setSeconds(0);
        date.setMilliseconds(0);
        const key = date.toISOString();
        
        if (!minuteData.has(key)) {
            minuteData.set(key, { count: 0, times: [] });
        }
        const data = minuteData.get(key);
        data.count++;
        data.times.push(row.total_time);
    });

    const trafficData = Array.from(minuteData.entries()).map(([key, data]) => {
        return {
            time: new Date(key),
            count: data.count,
            median: d3.median(data.times)
        };
    });
    trafficData.sort((a, b) => a.time - b.time);
    return trafficData;
}

function renderTrafficCharts() {
    d3.select("#requests-chart").selectAll("*").remove();
    d3.select("#latency-over-time-chart").selectAll("*").remove();

    if (rawData.length === 0) {
        d3.select("#requests-chart").append("div").text("No data available for the selected range.").style("padding", "20px");
        return;
    }

    const data = processTrafficData();
    if (data.length === 0) return;

    const margin = {top: 20, right: 30, bottom: 50, left: 60},
          width = document.getElementById('traffic-page').offsetWidth - margin.left - margin.right - 40,
          height = 300 - margin.top - margin.bottom;

    const x = d3.scaleTime()
        .domain(d3.extent(data, d => d.time))
        .range([0, width]);

    // Requests Chart
    const svg1 = d3.select("#requests-chart").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const y1 = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.count) * 1.1])
        .range([height, 0]);

    svg1.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x));
    svg1.append("g").call(d3.axisLeft(y1));

    const barWidth = Math.max(2, (width / data.length) * 0.8);

    svg1.selectAll(".bar")
        .data(data)
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", d => x(d.time) - barWidth/2)
        .attr("y", d => y1(d.count))
        .attr("width", barWidth)
        .attr("height", d => height - y1(d.count))
        .on("mouseover", (event, d) => {
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`Time: ${d.time.toLocaleString()}<br>Requests: ${d.count}`)
                .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));

    // Median Latency Chart
    const svg2 = d3.select("#latency-over-time-chart").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const y2 = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.median) * 1.2])
        .range([height, 0]);

    svg2.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x));
    svg2.append("g").call(d3.axisLeft(y2).tickFormat(d => d + "s"));

    const line = d3.line()
        .x(d => x(d.time))
        .y(d => y2(d.median))
        .curve(d3.curveMonotoneX);

    svg2.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("d", line);

    svg2.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", d => x(d.time))
        .attr("cy", d => y2(d.median))
        .attr("r", 3)
        .on("mouseover", (event, d) => {
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`Time: ${d.time.toLocaleString()}<br>Median Total Time: ${d.median.toFixed(3)}s`)
                .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));
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
    setRange(30);
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
    if (currentPage !== 'latency') return;
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

// Initial setup on load

// Initial range: 30 minutes
const now = new Date();
const start = new Date(now.getTime() - 30 * 60000);
document.getElementById('start-date').value = toISOStringLocal(start);
document.getElementById('end-date').value = toISOStringLocal(now);

// Initial fetch
fetchData();

// Auto refresh
setInterval(() => {
    if (document.getElementById('auto-refresh').checked) {
        fetchData();
    }
}, 30000);