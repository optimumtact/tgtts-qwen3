const DateTime = luxon.DateTime;
let availableVoices = [];
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
    
    fetchData();
}

async function fetchData() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    const params = new URLSearchParams();
    const start = DateTime.fromISO(startDate);
    const end = DateTime.fromISO(endDate);

    if (startDate) params.append('start_date', start.toUTC().toFormat('yyyy-MM-dd HH:mm:ss'));
    if (endDate) params.append('end_date', end.toUTC().toFormat('yyyy-MM-dd HH:mm:ss'));

    try {
        if (currentPage === 'latency') {
            // 1. Fetch voices to populate sidebar
            const vResponse = await fetch('/api/voices?' + params.toString());
            const vData = await vResponse.json();

            // 2. Fetch stats for selected voices (or all if none specified yet)
            let statsParams = new URLSearchParams(params);
            if (activeVoices.length > 0) {
                statsParams.append('voices', activeVoices.join(','));
            }
            const sResponse = await fetch('/api/ttsstats?' + statsParams.toString());
            const statsData = await sResponse.json();

            processData(statsData, vData);
            renderVoiceList();
            renderChart();
        } else {
            // Traffic page - calculate sensible period
            let period = 60;
            if (start.isValid && end.isValid) {
                const diffSeconds = end.diff(start, 'seconds').seconds;
                if (diffSeconds <= 3600) period = 60; // 1 min for up to 1 hour
                else if (diffSeconds <= 21600) period = 300; // 5 mins for up to 6 hours
                else if (diffSeconds <= 86400) period = 900; // 15 mins for up to 24 hours
                else if (diffSeconds <= 604800) period = 3600; // 1 hour for up to 7 days
                else if (diffSeconds <= 2592000) period = 21600; // 6 hours for up to 30 days
                else period = 86400; // 1 day for more than 30 days
            }

            params.append('period', period);
            let url = '/api/latency';
            if (params.toString()) url += '?' + params.toString();
            const response = await fetch(url);
            const data = await response.json();
            renderTrafficCharts(data);
        }
        
        document.getElementById('last-update').innerText = 'Last updated: ' + new Date().toLocaleTimeString();
    } catch (e) {
        console.error("Failed to fetch data", e);
    }
}

function processData(statsData, vData) {
    availableVoices = vData;
    voiceStats = [];
    globalStats = {};

    statsData.forEach(res => {
        const stats = {
            voice: res.voice,
            ...res.data,
            is_global: res.voice === 'global'
        };

        if (stats.is_global) {
            globalStats = stats;
        } else {
            voiceStats.push(stats);
        }
    });

    // Merge stats into availableVoices for the sidebar list metadata
    availableVoices.forEach(v => {
        const s = voiceStats.find(s => s.voice === v.voice);
        if (s) {
            v.p95 = s.p95;
            v.is_slow = globalStats.q3 ? s.p95 >= globalStats.q3 : false;
            v.is_fast = globalStats.q1 ? s.p95 <= globalStats.q1 : false;
            v.is_insignificant = s.count < 20;
        } else {
            v.is_insignificant = v.count < 20;
        }
    });
    
    // Sort sidebar voices by count
    availableVoices.sort((a, b) => b.count - a.count);
}

function renderTrafficCharts(data) {
    // (unchanged)
    const container = d3.select("#traffic-combined-chart");
    container.selectAll("*").remove();

    if (!data || data.length === 0) {
        container.append("div").text("No data available for the selected range.").style("padding", "20px");
        return;
    }

    data.forEach(d => {
        d.date = new Date(d.timestamp);
    });

    const margin = {top: 20, right: 60, bottom: 50, left: 60},
          width = document.getElementById('traffic-page').offsetWidth - margin.left - margin.right - 40,
          height = 400 - margin.top - margin.bottom;

    const svg = container.append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleTime()
        .domain(d3.extent(data, d => d.date))
        .range([0, width]);

    const yLat = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.p95) * 1.1 || 1])
        .range([height, 0]);

    const yCount = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.count) * 1.2 || 1])
        .range([height, 0]);

    svg.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x));
    svg.append("g").call(d3.axisLeft(yLat).tickFormat(d => d + "s"));
    svg.append("g").attr("transform", `translate(${width},0)`).call(d3.axisRight(yCount));

    svg.append("text").attr("transform", "rotate(-90)").attr("y", -margin.left + 15).attr("x", -height/2).attr("text-anchor", "middle").style("font-size", "12px").text("Latency (seconds)");
    svg.append("text").attr("transform", "rotate(-90)").attr("y", width + margin.right - 15).attr("x", -height/2).attr("text-anchor", "middle").style("font-size", "12px").text("Request Count");

    const area = d3.area()
        .x(d => x(d.date))
        .y0(d => yLat(d.min))
        .y1(d => yLat(d.p95))
        .curve(d3.curveMonotoneX);

    svg.append("path")
        .datum(data)
        .attr("fill", "#3498db")
        .attr("fill-opacity", 0.15)
        .attr("d", area);

    const lineMedian = d3.line()
        .x(d => x(d.date))
        .y(d => yLat(d.median))
        .curve(d3.curveMonotoneX);

    const lineMean = d3.line()
        .x(d => x(d.date))
        .y(d => yLat(d.mean))
        .curve(d3.curveMonotoneX);

    svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#3498db")
        .attr("stroke-width", 2)
        .attr("d", lineMedian);

    svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#2ecc71")
        .attr("stroke-width", 2)
        .style("stroke-dasharray", ("4, 4"))
        .attr("d", lineMean);

    svg.selectAll(".count-dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "count-dot")
        .attr("cx", d => x(d.date))
        .attr("cy", d => yCount(d.count))
        .attr("r", 4)
        .attr("fill", "#e74c3c")
        .attr("fill-opacity", 0.7)
        .on("mouseover", (event, d) => {
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`
                <b style="color:#e74c3c">Time: ${d.date.toLocaleString()}</b><hr>
                Requests: ${d.count}<br>
                Median Latency: ${d.median.toFixed(3)}s<br>
                Mean Latency: ${d.mean.toFixed(3)}s<br>
                P95 Latency: ${d.p95.toFixed(3)}s<br>
                Latency Range: ${d.min.toFixed(3)}s - ${d.max.toFixed(3)}s
            `)
            .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));
}

function renderVoiceList() {
    const list = d3.select("#voice-list");
    const term = document.getElementById('search-bar').value.toLowerCase();
    list.selectAll("*").remove();

    availableVoices.forEach(d => {
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
                fetchData();
            });
        
        item.append("span").attr("class", "v-name").text(`${d.voice} (${d.count})`);
        if (d.p95 !== undefined) {
            item.append("span").attr("class", "v-meta").text(`p95: ${d.p95.toFixed(2)}s`);
        }
        
        if(d.is_insignificant) item.append("span").attr("class", "badge orange-badge").text("LOW SAMPLES");
        if(d.is_slow) item.append("span").attr("class", "badge slow-badge").text("SLOW");
        if(d.is_fast) item.append("span").attr("class", "badge fast-badge").text("FAST");
    });
}

function filterVoices() {
    const term = document.getElementById('search-bar').value.toLowerCase();
    d3.selectAll(".voice-item").each(function() {
        const nameText = d3.select(this).select(".v-name").text().toLowerCase();
        const nameMatch = nameText.split(' (')[0].includes(term);
        d3.select(this).style("display", nameMatch ? "flex" : "none");
    });
}

function toggleAll(state) {
    if (state) {
        activeVoices = availableVoices.map(v => v.voice);
    } else {
        activeVoices = [];
    }
    fetchData();
}

function resetFilters() {
    setRange(30);
}

function renderChart() {
    if (currentPage !== 'latency') return;
    d3.select("#chart").selectAll("*").remove();
    
    if (!globalStats.voice && voiceStats.length === 0) {
        d3.select("#chart").append("div").text("No data available for the selected range.").style("padding", "20px");
        return;
    }

    const displayData = [];
    if (globalStats.voice) displayData.push(globalStats);
    displayData.push(...voiceStats);
    
    const margin = {top: 40, right: 30, bottom: 120, left: 60},
          width = Math.max(900, displayData.length * 100) - margin.left - margin.right,
          height = 500 - margin.top - margin.bottom;

    const svg = d3.select("#chart").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scalePoint().range([0, width]).domain(displayData.map(d => d.voice)).padding(0.5);
    const yMax = d3.max(displayData, d => d.max) || 1;
    const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([height, 0]);

    svg.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x))
        .selectAll("text").attr("transform", "rotate(-45)").style("text-anchor", "end");
    svg.append("g").call(d3.axisLeft(y).tickFormat(d => d + "s"));

    const violinWidth = 50;

    displayData.forEach(d => {
        if (d.count === 0) return;
        
        // Use pre-calculated KDE from API
        const density = d.kde ? d.kde.x.map((xVal, i) => [xVal, d.kde.y[i]]) : [];
        if (density.length === 0) return;

        const xPos = x(d.voice);
        const maxNum = d3.max(density, v => v[1]);
        const xNum = d3.scaleLinear().range([0, violinWidth / 2]).domain([0, maxNum]);
        
        const color = d.is_global ? "#9b59b6" : (d.count < 20 ? "#ffa500" : (globalStats.q3 && d.p95 >= globalStats.q3 ? "#e74c3c" : (globalStats.q1 && d.p95 <= globalStats.q1 ? "#2ecc71" : "#3498db")));

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
