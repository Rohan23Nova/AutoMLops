import json

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoMLOps Control Center</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://unpkg.com/feather-icons"></script>
  <style>
    :root {
      --bg: #09090B;
      --bg-sec: #0F1012;
      --card: #111214;
      --border: #23252A;
      --text-pri: #F4F4F5;
      --text-sec: #A1A1AA;
      --text-muted: #71717A;
      --accent: #6366f1; /* muted indigo */
      --success: #34d399; /* muted green */
      --warning: #fbbf24;
      --error: #f87171;
      --radius: 8px;
    }
    body {
      background-color: var(--bg);
      color: var(--text-pri);
      font-family: 'Inter', system-ui, sans-serif;
      margin: 0;
      padding: 0;
      font-size: 14px;
      line-height: 1.5;
      height: 100vh;
      overflow: hidden;
    }
    * {
      box-sizing: border-box;
    }
    
    /* Sidebar */
    .sidebar {
      width: 240px;
      background-color: var(--bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 24px 16px;
      flex-shrink: 0;
      z-index: 10;
    }
    .sidebar .logo {
      font-weight: 600;
      font-size: 16px;
      margin-bottom: 32px;
      padding-left: 12px;
      letter-spacing: -0.02em;
    }
    .nav {
      display: flex;
      flex-direction: column;
      gap: 4px;
      flex-grow: 1;
    }
    .nav a, .bottom-nav a {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 12px;
      color: var(--text-sec);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      border-radius: 6px;
      transition: all 0.2s;
      cursor: pointer;
    }
    .nav a:hover, .bottom-nav a:hover {
      background-color: rgba(255,255,255,0.03);
      color: var(--text-pri);
    }
    .nav a.active {
      background-color: rgba(255,255,255,0.05);
      color: var(--text-pri);
      box-shadow: inset 2px 0 0 var(--accent);
    }
    .nav a svg, .bottom-nav a svg {
      width: 16px;
      height: 16px;
    }
    .bottom-nav {
      display: flex;
      flex-direction: column;
      gap: 4px;
      border-top: 1px solid var(--border);
      padding-top: 16px;
      margin-top: 16px;
    }

    /* Content Area */
    .content {
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      background-color: var(--bg-sec);
      overflow: hidden;
    }
    .top-header {
      height: 64px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 32px;
      background-color: var(--bg);
      flex-shrink: 0;
    }
    .header-titles h1 {
      font-size: 16px;
      font-weight: 600;
      margin: 0;
      letter-spacing: -0.02em;
    }
    .header-titles p {
      font-size: 12px;
      color: var(--text-sec);
      margin: 2px 0 0 0;
    }
    .header-status {
      display: flex;
      align-items: center;
      gap: 24px;
      font-size: 12px;
      color: var(--text-sec);
    }
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: var(--text-muted);
    }
    .dot.success { background-color: var(--success); }
    .dot.warning { background-color: var(--warning); }
    .dot.error { background-color: var(--error); }
    
    .icon-btn {
      background: none;
      border: none;
      color: var(--text-sec);
      cursor: pointer;
      padding: 4px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: color 0.2s, background 0.2s;
    }
    .icon-btn:hover {
      color: var(--text-pri);
      background: rgba(255,255,255,0.05);
    }
    .icon-btn svg {
      width: 14px;
      height: 14px;
    }

    .dashboard-scroll-area {
      padding: 32px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 32px;
      scroll-behavior: smooth;
      flex-grow: 1;
    }

    /* Cards */
    .card {
      background-color: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      transition: background-color 0.2s;
    }
    .card:hover {
      background-color: #141517;
    }
    .card-title {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--text-sec);
      letter-spacing: 0.05em;
      margin: 0 0 16px 0;
      font-weight: 600;
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
    }
    .kpi-value {
      font-size: 24px;
      font-weight: 600;
      color: var(--text-pri);
      margin-bottom: 4px;
      letter-spacing: -0.02em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .kpi-sub {
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
    }

    /* Layout Grids */
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    .charts-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 24px;
    }
    .chart-container {
      position: relative;
      height: 200px;
      width: 100%;
    }

    /* Timelines */
    .timeline-horizontal {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 12px;
      color: var(--text-sec);
      margin-top: 24px;
      border-top: 1px solid var(--border);
      padding-top: 16px;
    }
    .timeline-item.active {
      color: var(--text-pri);
      font-weight: 500;
    }
    .timeline-arrow {
      color: var(--text-muted);
    }
    
    .timeline-vertical {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .tl-item {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 13px;
      color: var(--text-pri);
    }
    .tl-item.pending {
      color: var(--text-muted);
    }
    .tl-icon {
      font-size: 10px;
      width: 16px;
      text-align: center;
    }

    /* Forms */
    .form-group {
      margin-bottom: 16px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--text-sec);
      margin-bottom: 8px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    input[type="text"], input[type="number"], input[type="password"] {
      width: 100%;
      background-color: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-pri);
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-family: inherit;
      box-sizing: border-box;
      transition: border-color 0.2s;
    }
    input:focus {
      outline: none;
      border-color: var(--accent);
    }
    button {
      background-color: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    button:hover {
      opacity: 0.9;
    }
    button.btn-secondary {
      background-color: transparent;
      color: var(--text-pri);
      border: 1px solid var(--border);
    }
    button.btn-secondary:hover {
      background-color: rgba(255,255,255,0.05);
      opacity: 1;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Prediction Output */
    .prediction-result {
      margin-top: 24px;
      padding: 16px;
      background-color: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .prediction-result .val {
      font-size: 20px;
      font-weight: 600;
      color: var(--text-pri);
      margin-top: 4px;
    }

    /* Event Timeline */
    .event-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .event-item {
      display: flex;
      gap: 16px;
      font-size: 13px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      padding-bottom: 12px;
    }
    .event-item:last-child {
      border-bottom: none;
      padding-bottom: 0;
    }
    .event-time {
      color: var(--text-muted);
      font-variant-numeric: tabular-nums;
      min-width: 65px;
    }
    .event-body {
      color: var(--text-pri);
    }

    /* Services */
    .services-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .service-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 13px;
      color: var(--text-pri);
    }
    
    @media (max-width: 1024px) {
      .grid-2, .charts-grid { grid-template-columns: 1fr; }
      .sidebar { width: 64px; padding: 24px 8px; }
      .sidebar .logo, .nav a span, .bottom-nav a span { display: none; }
      .nav a { justify-content: center; padding: 12px; }
    }
  </style>
</head>
<body>

  <!-- LOGIN VIEW -->
  <div id="login-view" style="display: flex; flex-direction: column; height: 100vh; align-items: center; justify-content: center; background-color: var(--bg);">
      <div style="margin-bottom: 32px; text-align: center;">
          <div style="font-size: 28px; font-weight: 600; letter-spacing: -0.02em;">AutoMLOps</div>
          <div style="font-size: 14px; color: var(--text-sec); margin-top: 4px;">Control Center</div>
      </div>
      <div class="card" style="width: 360px;">
          <div class="form-group">
              <label>Username</label>
              <input type="text" id="login-user" placeholder="Enter username">
          </div>
          <div class="form-group">
              <label>Password</label>
              <input type="password" id="login-pass" placeholder="Enter password" onkeypress="if(event.key === 'Enter') login()">
          </div>
          <button style="width: 100%; margin-top: 8px;" onclick="login()">Sign In</button>
          <div id="login-msg" style="margin-top: 16px; font-size: 13px; text-align: center; font-weight: 500;"></div>
      </div>
  </div>
  
  <!-- MAIN DASHBOARD VIEW -->
  <div id="app-view" style="display: none; height: 100vh; width: 100%;">
      <!-- Sidebar -->
      <aside class="sidebar">
        <div class="logo">AutoMLOps</div>
        <nav class="nav">
           <a id="nav-overview" onclick="showTab('overview')" class="active"><i data-feather="grid"></i> <span>Overview</span></a>
           <a id="nav-predictions" onclick="showTab('predictions')"><i data-feather="activity"></i> <span>Predictions</span></a>
           <a id="nav-models" onclick="showTab('models')"><i data-feather="box"></i> <span>Models</span></a>
           <a id="nav-drift" onclick="showTab('drift')"><i data-feather="bar-chart-2"></i> <span>Data Drift</span></a>
           <a id="nav-retraining" onclick="showTab('retraining')"><i data-feather="refresh-cw"></i> <span>Retraining</span></a>
           <a id="nav-logs" onclick="showTab('logs')"><i data-feather="terminal"></i> <span>Logs</span></a>
        </nav>
        <div class="bottom-nav">
           <a onclick="logout()"><i data-feather="log-out"></i> <span>Logout</span></a>
        </div>
      </aside>
      
      <!-- Content -->
      <main class="content">
         <header class="top-header">
            <div class="header-titles">
               <h1>AutoMLOps Control Center</h1>
               <p>Machine Learning Operations & Observability</p>
            </div>
            <div class="header-status">
               <div class="status-indicator"><span class="dot success"></span> Operational</div>
               <div>Model <span id="hdr-model-version">vX</span></div>
               <div>Last updated: <span id="hdr-last-updated">0</span>s ago</div>
               <button class="icon-btn" onclick="refreshAll()"><i data-feather="refresh-cw"></i></button>
            </div>
         </header>
         
         <div class="dashboard-scroll-area">
            
            <!-- OVERVIEW TAB -->
            <div id="tab-overview" class="tab-content" style="display: flex; flex-direction: column; gap: 32px;">
                <div class="kpi-grid">
                   <div class="card">
                     <div class="card-title">Deployed Model</div>
                     <div class="kpi-value" id="kpi-model-name">-</div>
                     <div class="kpi-sub"><span class="dot success" style="margin-right:6px;"></span> Deployed</div>
                   </div>
                   <div class="card">
                     <div class="card-title">Predictions</div>
                     <div class="kpi-value" id="kpi-predictions">0</div>
                     <div class="kpi-sub">Total requests</div>
                   </div>
                   <div class="card">
                     <div class="card-title">Error Rate</div>
                     <div class="kpi-value" id="kpi-error-rate">0.0%</div>
                     <div class="kpi-sub" id="kpi-errors-count">0 errors</div>
                   </div>
                   <div class="card">
                     <div class="card-title">Avg Latency</div>
                     <div class="kpi-value" id="kpi-latency">0 ms</div>
                     <div class="kpi-sub">Overall avg</div>
                   </div>
                   <div class="card">
                     <div class="card-title">Batch Jobs</div>
                     <div class="kpi-value" id="kpi-batch">0</div>
                     <div class="kpi-sub">Completed</div>
                   </div>
                   <div class="card">
                     <div class="card-title">API Status</div>
                     <div class="kpi-value" id="kpi-api-status" style="color:var(--success);">200 OK</div>
                     <div class="kpi-sub">Operational</div>
                   </div>
                </div>

                <div class="charts-grid">
                   <div class="card">
                      <div class="card-title">Prediction Traffic</div>
                      <div class="chart-container">
                        <canvas id="trafficChart"></canvas>
                      </div>
                   </div>
                   <div class="card">
                      <div class="card-title">Prediction Latency</div>
                      <div class="chart-container">
                        <canvas id="latencyChart"></canvas>
                      </div>
                   </div>
                </div>

                <div id="overview-deployment-container">
                    <!-- card-deployment is injected here on overview load -->
                </div>
                
                <div id="overview-drift-retrain-grid" class="grid-2">
                    <!-- card-drift and card-retraining are injected here -->
                </div>

                <div id="overview-health-logs-grid" class="grid-2">
                   <div class="card" id="card-health">
                     <div class="card-title">System Health</div>
                     <div class="services-list">
                        <div class="service-item">
                           <span>FastAPI</span>
                           <span style="display:flex; align-items:center; gap:6px;"><span class="dot success"></span>Operational</span>
                        </div>
                        <div class="service-item">
                           <span>Prometheus</span>
                           <span style="display:flex; align-items:center; gap:6px;"><span class="dot success"></span>Operational</span>
                        </div>
                        <div class="service-item">
                           <span>MLflow</span>
                           <span style="display:flex; align-items:center; gap:6px;"><span class="dot success"></span>Operational</span>
                        </div>
                        <div class="service-item">
                           <span>Airflow</span>
                           <span style="display:flex; align-items:center; gap:6px;"><span class="dot success"></span>Operational</span>
                        </div>
                        <div class="service-item">
                           <span>Model Serving</span>
                           <span style="display:flex; align-items:center; gap:6px;"><span class="dot success"></span>Operational</span>
                        </div>
                     </div>
                   </div>
                   <!-- card-logs is injected here -->
                </div>
            </div>

            <!-- PREDICTIONS TAB -->
            <div id="tab-predictions" class="tab-content" style="display: none; flex-direction: column; gap: 32px;">
                <div class="grid-2" id="grid-predictions">
                   <div class="card">
                     <div class="card-title">Run Prediction</div>
                     <form id="predict-form">
                        <div class="grid-2" style="gap:16px;">
                          <div class="form-group">
                            <label>Age</label>
                            <input type="number" id="pred-age" placeholder="e.g. 28" required>
                          </div>
                          <div class="form-group">
                            <label>Income</label>
                            <input type="number" id="pred-income" placeholder="e.g. 50000" required>
                          </div>
                          <div class="form-group">
                            <label>City</label>
                            <input type="text" id="pred-city" placeholder="e.g. New York" required>
                          </div>
                          <div class="form-group">
                            <label>Student</label>
                            <input type="text" id="pred-student" placeholder="Yes / No" required>
                          </div>
                        </div>
                        <button type="submit">Run Prediction</button>
                     </form>
                     
                     <div class="prediction-result" id="pred-result" style="display:none;">
                        <div>
                          <div style="font-size:12px; color:var(--text-sec); text-transform:uppercase; letter-spacing:0.05em;">Prediction</div>
                          <div class="val" id="pred-output">-</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:12px; color:var(--text-sec); text-transform:uppercase; letter-spacing:0.05em;">Latency</div>
                          <div class="val" id="pred-latency" style="font-size:16px; font-weight:500;">-</div>
                        </div>
                     </div>
                   </div>
                   
                   <div class="card">
                     <div class="card-title">Batch Inference</div>
                     <p style="font-size:13px; color:var(--text-sec); margin-bottom:24px;">Upload a CSV file for processing multiple records through the model.</p>
                     
                     <div style="border:1px dashed var(--border); border-radius:6px; padding:32px; text-align:center; margin-bottom:24px;">
                       <input type="file" id="batch-file" accept=".csv" style="display:none;" onchange="document.getElementById('file-name').innerText = this.files[0].name">
                       <button class="btn-secondary" onclick="document.getElementById('batch-file').click()">Browse files</button>
                       <div id="file-name" style="margin-top:12px; font-size:12px; color:var(--text-sec);">No file selected</div>
                     </div>
                     
                     <button onclick="runBatch()">Run Batch</button>
                     
                     <div class="prediction-result" id="batch-result" style="display:none;">
                        <div>
                          <div style="font-size:12px; color:var(--text-sec); text-transform:uppercase; letter-spacing:0.05em;">Status</div>
                          <div class="val" style="font-size:16px; color:var(--success);">Completed</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:12px; color:var(--text-sec); text-transform:uppercase; letter-spacing:0.05em;">Records</div>
                          <div class="val" id="batch-records" style="font-size:16px; font-weight:500;">-</div>
                        </div>
                     </div>
                   </div>
                </div>
            </div>
            
            <!-- MODELS TAB -->
            <div id="tab-models" class="tab-content" style="display: none; flex-direction: column; gap: 32px;"></div>

            <!-- DRIFT TAB -->
            <div id="tab-drift" class="tab-content" style="display: none; flex-direction: column; gap: 32px;"></div>

            <!-- RETRAINING TAB -->
            <div id="tab-retraining" class="tab-content" style="display: none; flex-direction: column; gap: 32px;"></div>
            
            <!-- LOGS TAB -->
            <div id="tab-logs" class="tab-content" style="display: none; flex-direction: column; gap: 32px;"></div>
            
         </div>
      </main>
  </div>

  <!-- PORTABLE DOM ELEMENTS (Moved dynamically by tab clicks) -->
  <div style="display:none;" id="portable-elements-container">
      <div id="card-deployment" class="card">
         <div class="card-title">Deployment</div>
         <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:24px;">
           <div style="overflow: hidden;">
             <div class="kpi-sub" style="margin-bottom:4px; font-weight:500;">MODEL</div>
             <div style="font-size:14px; font-weight:500; color:var(--text-pri); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" id="dep-model-name">-</div>
           </div>
           <div style="overflow: hidden;">
             <div class="kpi-sub" style="margin-bottom:4px; font-weight:500;">VERSION</div>
             <div style="font-size:14px; font-weight:500; color:var(--text-pri);" id="dep-model-version">-</div>
           </div>
           <div>
             <div class="kpi-sub" style="margin-bottom:4px; font-weight:500;">STATUS</div>
             <div style="font-size:14px; font-weight:500; color:var(--text-pri); display:flex; align-items:center; gap:6px;">
               <span class="dot success"></span>Production
             </div>
           </div>
           <div>
             <div class="kpi-sub" style="margin-bottom:4px; font-weight:500;">SOURCE</div>
             <div style="font-size:14px; font-weight:500; color:var(--text-pri);">MLflow Model Registry</div>
           </div>
         </div>
         <div class="timeline-horizontal">
            <div class="timeline-item">Training</div>
            <div class="timeline-arrow">→</div>
            <div class="timeline-item">Evaluation</div>
            <div class="timeline-arrow">→</div>
            <div class="timeline-item">Registry</div>
            <div class="timeline-arrow">→</div>
            <div class="timeline-item active" style="color:var(--accent);">Deployment</div>
         </div>
      </div>

      <div id="card-drift" class="card">
         <div class="card-title">Data Drift</div>
         <div style="font-size:13px; color:var(--text-sec); margin-bottom:24px;">Distribution changes between reference and current data.</div>
         <div style="display:flex; align-items:center; gap:8px; margin-bottom:24px;">
            <span class="dot success" id="drift-dot"></span>
            <span style="font-size:14px; font-weight:500; color:var(--text-pri);" id="drift-status">No significant drift</span>
         </div>
         <div style="font-size:12px; color:var(--text-muted); margin-bottom:24px; line-height:1.6;" id="drift-details">
            Last checked: Not available<br>
            Features checked: Not available
         </div>
         <button class="btn-secondary" onclick="runDriftCheck()">Run Drift Check</button>
      </div>

      <div id="card-retraining" class="card">
         <div class="card-title">Retraining Pipeline</div>
         <div class="timeline-vertical">
            <div class="tl-item"><span class="tl-icon">✓</span> Drift Check</div>
            <div class="tl-item"><span class="tl-icon">✓</span> Decision</div>
            <div class="tl-item pending"><span class="tl-icon">●</span> Training</div>
            <div class="tl-item pending"><span class="tl-icon">●</span> Evaluation</div>
            <div class="tl-item pending"><span class="tl-icon">●</span> Registry</div>
            <div class="tl-item pending"><span class="tl-icon">●</span> Deployment</div>
         </div>
         <button class="btn-secondary" style="margin-top:24px;" onclick="triggerRetrain()">Trigger Retraining</button>
      </div>

      <div id="card-logs" class="card">
         <div class="card-title">Recent Events</div>
         <div class="event-list" id="events-list">
            <div style="font-size:13px; color:var(--text-sec);">No events yet</div>
         </div>
      </div>
  </div>
  
  <script>
    feather.replace();

    const API_BASE = "http://localhost:8000";
    let token = localStorage.getItem('automlops_token') || "";

    function showTab(tab) {
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        document.getElementById('tab-' + tab).style.display = 'flex';
        
        document.querySelectorAll('.nav a').forEach(el => el.classList.remove('active'));
        document.getElementById('nav-' + tab).classList.add('active');
        
        // Dynamic Card Routing
        const depCard = document.getElementById('card-deployment');
        const driftCard = document.getElementById('card-drift');
        const retrainCard = document.getElementById('card-retraining');
        const logsCard = document.getElementById('card-logs');

        if (tab === 'overview') {
            document.getElementById('overview-deployment-container').appendChild(depCard);
            document.getElementById('overview-drift-retrain-grid').appendChild(driftCard);
            document.getElementById('overview-drift-retrain-grid').appendChild(retrainCard);
            document.getElementById('overview-health-logs-grid').appendChild(logsCard);
        } else {
            if (tab === 'models') document.getElementById('tab-models').appendChild(depCard);
            if (tab === 'drift') document.getElementById('tab-drift').appendChild(driftCard);
            if (tab === 'retraining') document.getElementById('tab-retraining').appendChild(retrainCard);
            if (tab === 'logs') document.getElementById('tab-logs').appendChild(logsCard);
        }
    }

    function checkAuth() {
        if (token) {
            document.getElementById('login-view').style.display = 'none';
            document.getElementById('app-view').style.display = 'flex';
            showTab('overview');
            initCharts();
            refreshAll();
        } else {
            document.getElementById('login-view').style.display = 'flex';
            document.getElementById('app-view').style.display = 'none';
        }
    }

    async function login() {
        const u = document.getElementById('login-user').value;
        const p = document.getElementById('login-pass').value;
        const msg = document.getElementById('login-msg');
        
        msg.innerText = "Authenticating...";
        msg.style.color = "var(--text-sec)";
        
        try {
            const formData = new URLSearchParams();
            formData.append("username", u);
            formData.append("password", p);
            
            const res = await fetch(API_BASE + "/login", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: formData
            });
            
            if (res.ok) {
                const data = await res.json();
                token = data.access_token || data.token;
                localStorage.setItem('automlops_token', token);
                checkAuth();
            } else {
                msg.innerText = "Invalid credentials.";
                msg.style.color = "var(--error)";
            }
        } catch (e) {
            msg.innerText = "Connection error.";
            msg.style.color = "var(--error)";
        }
    }

    function logout() {
        token = "";
        localStorage.removeItem('automlops_token');
        checkAuth();
    }

    // Chart Setup
    let chartTimeLabels = [];
    let chartTraffic = [];
    let chartLatency = [];
    let trafficChart, latencyChart;
    let chartsInitialized = false;

    function initCharts() {
        if (chartsInitialized) return;
        const commonOpt = {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#23252A', drawBorder: false }, ticks: { color: '#71717A', maxTicksLimit: 6 } },
                y: { grid: { color: '#23252A', drawBorder: false }, ticks: { color: '#71717A', maxTicksLimit: 5 }, beginAtZero: true }
            }
        };
        
        trafficChart = new Chart(document.getElementById('trafficChart'), {
            type: 'line',
            data: { labels: chartTimeLabels, datasets: [{ data: chartTraffic, borderColor: '#6366f1', borderWidth: 1.5, tension: 0.1, pointRadius: 0 }] },
            options: commonOpt
        });
        
        latencyChart = new Chart(document.getElementById('latencyChart'), {
            type: 'line',
            data: { labels: chartTimeLabels, datasets: [{ data: chartLatency, borderColor: '#6366f1', borderWidth: 1.5, tension: 0.1, pointRadius: 0 }] },
            options: commonOpt
        });
        chartsInitialized = true;
    }

    function parsePrometheus(text) {
        const lines = text.split('\\n');
        const metrics = {};
        for(let line of lines) {
            if(!line || line.startsWith('#')) continue;
            const parts = line.split(' ');
            const key = parts[0].split('{')[0];
            const val = parseFloat(parts[1]);
            if(key.endsWith('_sum') || key.endsWith('_count')) {
                 metrics[key] = (metrics[key] || 0) + val;
            } else {
                 metrics[key] = val;
            }
        }
        return metrics;
    }

    let lastReqCount = 0;

    async function fetchMetrics() {
        if(!token) return;
        try {
            const res = await fetch(API_BASE + '/metrics');
            if(!res.ok) throw new Error();
            const text = await res.text();
            const m = parsePrometheus(text);
            
            const preds = m['automlops_predictions_total'] || 0;
            const errs = m['automlops_prediction_errors_total'] || 0;
            const batches = m['automlops_batch_predictions_total'] || 0;
            
            const latSum = m['automlops_prediction_latency_seconds_sum'] || 0;
            const latCount = m['automlops_prediction_latency_seconds_count'] || 0;
            const avgLat = latCount > 0 ? (latSum / latCount) : 0;
            
            const errRate = preds > 0 ? ((errs / preds) * 100).toFixed(1) : '0.0';
            
            document.getElementById('kpi-predictions').innerText = preds.toLocaleString();
            document.getElementById('kpi-error-rate').innerText = errRate + '%';
            document.getElementById('kpi-errors-count').innerText = errs + ' errors';
            document.getElementById('kpi-latency').innerText = (avgLat * 1000).toFixed(0) + ' ms';
            document.getElementById('kpi-batch').innerText = batches.toLocaleString();
            
            document.getElementById('kpi-api-status').innerText = '200 OK';
            document.getElementById('kpi-api-status').style.color = 'var(--success)';
            
            const now = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
            chartTimeLabels.push(now);
            
            const trafficDiff = Math.max(0, preds - lastReqCount);
            if(lastReqCount === 0 && trafficDiff === preds && preds > 0) {
               chartTraffic.push(0); 
            } else {
               chartTraffic.push(trafficDiff);
            }
            lastReqCount = preds;
            
            chartLatency.push(avgLat * 1000);
            
            if(chartTimeLabels.length > 20) {
                chartTimeLabels.shift();
                chartTraffic.shift();
                chartLatency.shift();
            }
            
            if (chartsInitialized) {
                trafficChart.update();
                latencyChart.update();
            }
            
            document.getElementById('hdr-last-updated').innerText = '0';
        } catch(e) {
            document.getElementById('kpi-api-status').innerText = 'Failing';
            document.getElementById('kpi-api-status').style.color = 'var(--error)';
        }
    }

    async function fetchModelInfo() {
        if(!token) return;
        try {
            const res = await fetch(API_BASE + '/model-info');
            if(res.ok) {
                const data = await res.json();
                let name = data.model_name || 'AutoMLOps';
                name = name.replace(/_model/i, '');
                const ver = data.model_version || 'X';
                document.getElementById('kpi-model-name').innerText = name;
                document.getElementById('dep-model-name').innerText = name;
                document.getElementById('hdr-model-version').innerText = 'v' + ver;
                document.getElementById('dep-model-version').innerText = 'v' + ver;
            }
        } catch(e) {}
    }

    async function fetchLogs() {
        if(!token) return;
        try {
            const res = await fetch(API_BASE + '/logs', {
                headers: { "Authorization": `Bearer ${token}` }
            });
            if(res.status === 401) { logout(); return; }
            if(res.ok) {
                const data = await res.json();
                const logs = data.logs || [];
                const list = document.getElementById('events-list');
                
                if(logs.length === 0) return;
                list.innerHTML = '';
                
                logs.slice(-6).reverse().forEach(log => {
                    let timeStr = "--:--:--";
                    let msgStr = log;
                    
                    const match = log.match(/^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2})/);
                    if(match) {
                        timeStr = match[1].split(' ')[1];
                        msgStr = log.substring(match[0].length).trim();
                        msgStr = msgStr.replace(/^[\\|\\-\\s]+/, '').replace(/^(INFO|ERROR|WARNING):?\\s*/i, '');
                    }
                    
                    const item = document.createElement('div');
                    item.className = 'event-item';
                    item.innerHTML = `
                      <div class="event-time">${timeStr}</div>
                      <div class="event-body">${msgStr}</div>
                    `;
                    list.appendChild(item);
                });
            }
        } catch(e) {}
    }

    // Tools Handlers
    document.getElementById('predict-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            data: {
                age: parseFloat(document.getElementById('pred-age').value),
                income: parseFloat(document.getElementById('pred-income').value),
                city: document.getElementById('pred-city').value,
                student: document.getElementById('pred-student').value
            }
        };
        
        const btn = e.target.querySelector('button');
        btn.innerText = 'Running...';
        btn.disabled = true;
        
        const start = performance.now();
        try {
            const res = await fetch(API_BASE + '/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify(payload)
            });
            if(res.status === 401) { logout(); return; }
            
            const end = performance.now();
            const lat = Math.round(end - start);
            
            if(res.ok) {
                const data = await res.json();
                document.getElementById('pred-result').style.display = 'flex';
                document.getElementById('pred-output').innerText = (data.prediction !== undefined ? data.prediction : data.class_name) || "Completed";
                document.getElementById('pred-latency').innerText = lat + ' ms';
                setTimeout(fetchMetrics, 500);
            } else {
                alert('Prediction failed: ' + res.status);
            }
        } catch(e) {
            alert('Network error.');
        } finally {
            btn.innerText = 'Run Prediction';
            btn.disabled = false;
        }
    });

    async function runBatch() {
        const file = document.getElementById('batch-file').files[0];
        if(!file) { alert('Select a CSV file.'); return; }
        
        const btn = document.querySelector('#tab-predictions .card:nth-child(2) button:not(.btn-secondary)');
        btn.innerText = 'Processing...';
        btn.disabled = true;
        
        const fd = new FormData();
        fd.append('file', file);
        
        try {
            const res = await fetch(API_BASE + '/batch_predict', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: fd
            });
            if(res.status === 401) { logout(); return; }
            
            if(res.ok) {
                const data = await res.json();
                document.getElementById('batch-result').style.display = 'flex';
                document.getElementById('batch-records').innerText = data.rows_received || 'Done';
                setTimeout(fetchMetrics, 500);
            } else {
                alert('Batch processing failed.');
            }
        } catch(e) {
            alert('Network error.');
        } finally {
            btn.innerText = 'Run Batch';
            btn.disabled = false;
        }
    }

    async function runDriftCheck() {
        const btn = document.querySelector('#card-drift button');
        btn.innerText = 'Checking...';
        btn.disabled = true;
        
        try {
            const res = await fetch(API_BASE + '/check-drift', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if(res.status === 401) { logout(); return; }
            if(res.ok) {
                const data = await res.json();
                const isDrift = data.drift_detected === true;
                
                document.getElementById('drift-dot').className = 'dot ' + (isDrift ? 'error' : 'success');
                document.getElementById('drift-status').innerText = isDrift ? 'Drift Detected' : 'No significant drift';
                document.getElementById('drift-status').style.color = isDrift ? 'var(--error)' : 'var(--text-pri)';
                
                const now = new Date().toLocaleTimeString();
                document.getElementById('drift-details').innerHTML = `Last checked: ${now}<br>Status: ${data.message || 'Complete'}`;
            }
        } catch(e) {
            alert('Drift check failed.');
        } finally {
            btn.innerText = 'Run Drift Check';
            btn.disabled = false;
        }
    }

    async function triggerRetrain() {
        const btn = document.querySelector('#card-retraining button');
        btn.innerText = 'Triggering...';
        btn.disabled = true;
        
        try {
            const res = await fetch(API_BASE + '/retrain', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if(res.status === 401) { logout(); return; }
            if(res.ok) {
                alert('Retraining pipeline triggered successfully.');
            }
        } catch(e) {
            alert('Failed to trigger retraining.');
        } finally {
            btn.innerText = 'Trigger Retraining';
            btn.disabled = false;
        }
    }

    function refreshAll() {
        fetchModelInfo();
        fetchMetrics();
        fetchLogs();
    }

    // Timers
    setInterval(() => {
        if(!token) return;
        const el = document.getElementById('hdr-last-updated');
        el.innerText = parseInt(el.innerText) + 1;
    }, 1000);

    setInterval(() => {
        if(!token) return;
        fetchMetrics();
        fetchLogs();
    }, 5000);

    window.onload = () => {
        checkAuth();
    };
  </script>
</body>
</html>
"""

with open("index.html", "w") as f:
    f.write(html_content)

