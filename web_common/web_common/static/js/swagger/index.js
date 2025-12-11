const csrfToken = document.head.querySelector('meta[name="csrf"]')?.content || '';
const basePath = document.head.querySelector('meta[name="base-path"]')?.content || '';

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
