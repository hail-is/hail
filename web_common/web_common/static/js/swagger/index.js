const csrfToken = document.getElementById('_csrf').value;
const basePath = document.getElementById('_base_path').value;

window.onload = function() {
    window.ui = SwaggerUIBundle({
        url: `${basePath}/openapi.yaml`,
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIStandalonePreset
        ],
        plugins: [
            SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout",
        requestInterceptor: function(request) {
            if (csrfToken) {
                // Add CSRF token to request headers
                if (!request.headers) {
                    request.headers = {};
                }
                request.headers['X-CSRF-Token'] = csrfToken;
            }

            return request;
        }
    });
};
