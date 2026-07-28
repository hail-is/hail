import { createRoot } from 'react-dom/client';
import SwaggerUI from 'swagger-ui-react';
import 'swagger-ui-react/swagger-ui.css';

const basePath = document.head.querySelector('meta[name="base-path"]')?.getAttribute('value') ?? '';
const csrfToken = document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') ?? '';
const rootEl = document.getElementById('swagger-ui');

if (rootEl) {
  createRoot(rootEl).render(
    <SwaggerUI
      url={`${basePath}/openapi.yaml`}
      deepLinking={true}
      requestInterceptor={(request) => {
        if (csrfToken) {
          if (!request.headers) {
            request.headers = {};
          }
          request.headers['X-CSRF-Token'] = csrfToken;
        }
        return request;
      }}
    />,
  );
}
