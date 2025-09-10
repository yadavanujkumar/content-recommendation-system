"""
Simple web interface for testing the recommendation API
"""

import http.server
import socketserver
import json
import urllib.request
import urllib.parse
from urllib.parse import urlparse, parse_qs

class RecommendationHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        if path == '/':
            self.serve_main_page()
        elif path == '/api/recommendations':
            self.handle_recommendations(query_params)
        elif path == '/api/health':
            self.handle_health_check()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Content Recommendation System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #1f77b4; margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 200px; }
        button { background: #1f77b4; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1565c0; }
        .recommendations { margin-top: 30px; }
        .rec-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 4px; background: #fafafa; }
        .rec-title { font-weight: bold; color: #333; }
        .rec-meta { color: #666; font-size: 0.9em; margin: 5px 0; }
        .rec-score { color: #1f77b4; font-weight: bold; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .status.success { background: #e8f5e8; color: #2e7d32; }
        .status.error { background: #ffebee; color: #c62828; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header">🎯 Content Recommendation System</h1>
        
        <div id="status"></div>
        
        <form id="recommendationForm">
            <div class="form-group">
                <label for="userId">User ID (0-99):</label>
                <input type="number" id="userId" name="userId" value="0" min="0" max="99">
            </div>
            
            <div class="form-group">
                <label for="method">Recommendation Method:</label>
                <select id="method" name="method">
                    <option value="hybrid">Hybrid</option>
                    <option value="content">Content-based</option>
                    <option value="collaborative">Collaborative</option>
                    <option value="popular">Popular</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="k">Number of recommendations:</label>
                <input type="number" id="k" name="k" value="5" min="1" max="20">
            </div>
            
            <div class="form-group">
                <label for="category">Category (optional):</label>
                <select id="category" name="category">
                    <option value="">All categories</option>
                    <option value="technology">Technology</option>
                    <option value="sports">Sports</option>
                    <option value="entertainment">Entertainment</option>
                    <option value="science">Science</option>
                    <option value="politics">Politics</option>
                    <option value="health">Health</option>
                    <option value="travel">Travel</option>
                    <option value="food">Food</option>
                </select>
            </div>
            
            <button type="submit">Get Recommendations</button>
        </form>
        
        <div id="recommendations" class="recommendations"></div>
    </div>

    <script>
        document.getElementById('recommendationForm').addEventListener('submit', function(e) {
            e.preventDefault();
            getRecommendations();
        });
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function getRecommendations() {
            const formData = new FormData(document.getElementById('recommendationForm'));
            const userId = formData.get('userId');
            const method = formData.get('method');
            const k = formData.get('k');
            const category = formData.get('category');
            
            let url = `/api/recommendations?user_id=${userId}&method=${method}&k=${k}`;
            if (category) {
                url += `&category=${category}`;
            }
            
            showStatus('Getting recommendations...', 'success');
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showStatus(`Error: ${data.error}`, 'error');
                        return;
                    }
                    
                    displayRecommendations(data);
                    showStatus(`Got ${data.recommendations.length} recommendations using ${data.method} method`, 'success');
                })
                .catch(error => {
                    showStatus(`Error: ${error.message}`, 'error');
                });
        }
        
        function displayRecommendations(data) {
            const recDiv = document.getElementById('recommendations');
            
            if (!data.recommendations || data.recommendations.length === 0) {
                recDiv.innerHTML = '<p>No recommendations found.</p>';
                return;
            }
            
            let html = `<h2>Recommendations for User ${data.user_id}</h2>`;
            
            data.recommendations.forEach((rec, index) => {
                html += `
                    <div class="rec-item">
                        <div class="rec-title">${index + 1}. ${rec.title}</div>
                        <div class="rec-meta">Category: ${rec.category} | Type: ${rec.content_type} | Score: <span class="rec-score">${rec.score}</span></div>
                        <div class="rec-meta">Author: ${rec.author} | Created: ${rec.created_date}</div>
                        <p>${rec.description}</p>
                        ${rec.reason ? `<div class="rec-meta"><em>${rec.reason}</em></div>` : ''}
                    </div>
                `;
            });
            
            recDiv.innerHTML = html;
        }
        
        // Check API health on page load
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'healthy') {
                    showStatus(`✅ API Connected - ${data.data_loaded.users} users, ${data.data_loaded.items} items, ${data.data_loaded.interactions} interactions`, 'success');
                } else {
                    showStatus('❌ API not healthy', 'error');
                }
            })
            .catch(error => {
                showStatus('❌ Cannot connect to API', 'error');
            });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def handle_recommendations(self, query_params):
        try:
            user_id = query_params.get('user_id', ['0'])[0]
            method = query_params.get('method', ['hybrid'])[0]
            k = query_params.get('k', ['5'])[0]
            category = query_params.get('category', [''])[0]
            
            # Build URL for backend API
            api_url = f"http://localhost:8000/recommendations/{user_id}?k={k}&method={method}"
            if category:
                api_url += f"&category={category}"
            
            # Make request to backend API
            with urllib.request.urlopen(api_url) as response:
                data = json.loads(response.read())
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def handle_health_check(self):
        try:
            with urllib.request.urlopen("http://localhost:8000/health") as response:
                data = json.loads(response.read())
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

def run_server(port=8080):
    with socketserver.TCPServer(("", port), RecommendationHandler) as httpd:
        print(f"Frontend server running on http://localhost:{port}")
        print("Make sure the API server is running on http://localhost:8000")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()