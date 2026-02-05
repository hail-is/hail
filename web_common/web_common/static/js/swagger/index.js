const csrfToken = document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') || '';
const basePath = document.head.querySelector('meta[name="base-path"]')?.getAttribute('value') || '';

if (Object.hasOwn(window, "SwaggerUIBundle") && Object.hasOwn(window, "SwaggerUIStandalonePreset")) {
    window.onload = function() {
        window.ui = window.SwaggerUIBundle({
            url: `${basePath}/openapi.yaml`,
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
                window.SwaggerUIBundle.presets.apis,
                window.SwaggerUIStandalonePreset
            ],
            plugins: [
                window.SwaggerUIBundle.plugins.DownloadUrl
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
}
