window.onload = function() {
  //<editor-fold desc="Changeable Configuration Block">

  // the following lines will be replaced by docker/configurator, when it runs in a docker-container
  window.ui = SwaggerUIBundle({
    url: '{{ base_path }}/openapi.yaml',
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
      // Get CSRF token from template variable
      const csrfToken = "{{ csrf_token }}";
      
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
  //</editor-fold>
};
