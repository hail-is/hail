# `services/ui` — Shared Frontend

> [!NOTE]
> This document covers React components in `services/ui`. The rest of Hail's service UIs — Jinja2 templates, static assets, vanilla JS — are not separately documented yet; the [Background](#background-how-hail-uis-normally-work) section below gives a brief overview of that pattern.

## Background: how Hail UIs normally work

Most UI pages in Hail follow a simple server-side pattern:

1. A browser request hits a service endpoint (e.g. `GET /batches` in the batch service).
2. The handler fetches whatever data it needs from the database and builds a Python context dict.
3. The handler calls `render_template`, which renders a Jinja2 HTML template using that context and returns the page.

Static assets (CSS, images, the occasional vanilla JS snippet) live inside the service directory and are served from a static route. This is fast to develop, easy to reason about, and works well for most pages.

## React 101

React is a JavaScript library for building UIs out of components — functions that accept data as props and return a description of what to render. The files here are written in TSX (TypeScript + JSX), a syntax that lets you write HTML-like markup directly inside TypeScript. esbuild compiles `.tsx` to a plain `.js` bundle that the browser can run.

### Components and state

A component re-renders whenever its state changes. State is declared with `useState`:

```tsx
const [data, setData] = useState<Row[]>([]);
```

`data` is the current value; `setData` is the setter. Calling the setter schedules a re-render with the new value. State is local to the component instance — sibling components have their own copies.

### useEffect and data fetching

`useEffect` runs side effects after a render. The most common use here is fetching data from an API on mount:

```tsx
useEffect(() => {
  fetch('/api/v1alpha/some_endpoint')
    .then(r => r.json())
    .then(json => setData(json.rows));
}, []);
```

The second argument is the **dependency array**. An empty array `[]` means "run once after the first render". If you include values — `[filter, page]` — the effect re-runs whenever any of them changes.

**Things to watch out for:**

- **Missing dependencies.** If the effect uses a variable but it's not in the dependency array, the effect captures a stale version of it. TypeScript won't catch this; the `react-hooks/exhaustive-deps` ESLint rule would, but we don't currently run it. Read effects carefully.
- **Infinite loops.** If you call a setter inside an effect and include that state in the dependency array, you get: render → effect → setState → render → effect → … . Keep setters out of the dependency array (they're stable references and don't need to be listed).
- **Cleanup.** If a component unmounts while a fetch is in flight, setting state on it causes a warning. Return a cleanup function that signals cancellation, or check a `mounted` flag before calling the setter.
- **Effects run after paint.** The browser renders first, then effects fire. For data fetching this means there's always at least one render with the initial (empty) state — handle loading states explicitly rather than assuming data is present.

### The render model

React diffs a virtual DOM against the real DOM and applies only the changes. This is efficient but means object identity matters for performance — recreating objects or arrays on every render can cause unnecessary child re-renders. For the scale of pages here this rarely matters, but it's worth knowing if something seems to re-render unexpectedly.

## Why React components

- **Stateful interactivity.** Some UI interactions can't be handled server-side: paginated tables with live filters, charts that update as the user adjusts a time range, rows that expand to show nested detail. These are straightforward in React and messy in vanilla JS.
- **Richer components.** The npm ecosystem has high-quality, well-maintained libraries for things that are tedious to build from scratch — charting, data grids, date pickers, etc. esbuild bundles only what's used, so pulling in a library doesn't bloat unrelated pages.
- **Easier to compose functionality.** Hooks and component composition make it straightforward to share logic across a page — data fetching, filtering state, pagination — without resorting to global variables or duplicating code across template partials.
- **Separation of data from layout.** In the Jinja2 model, the Python handler must fetch all data before the page can render. A React component fetches from an API endpoint independently, which means the page shell loads immediately and data arrives asynchronously. It also means the API endpoint is a clean, testable contract: the component doesn't care how the data is produced, only what shape it has.
- **Mock data in development.** Because the component talks to an API rather than receiving data baked into the HTML, the dev-proxy can intercept those API calls and return fixture data. The full page experience is reproducible locally without a deployed service or live database. See the [dev-proxy section](#local-development-with-the-dev-proxy) below.
- **Build-time type safety.** TypeScript catches shape mismatches between what the API returns and what the component expects before anything is deployed. `npm run check` runs `tsc --noEmit` as part of the build pipeline.

## Gradual rollout

Rather than rewriting all service UIs in React, the approach here is incremental:

- Jinja2 templates remain the default. New pages that are mostly static continue to use them.
- For pages or components where the above advantages are worth it, a React component is added to `services/ui/` and compiled to a JS bundle that the template loads.
- The template still controls the page shell, navigation, and initial data; React takes over a specific DOM node to render its part of the page.

This keeps the blast radius of frontend changes small. A change to one React component doesn't touch the Jinja2 layer, and vice versa.

## Structure

```
services/ui/
  package.json          # "hail-ui" — esbuild + TypeScript dev deps
  package-lock.json
  tsconfig.json
  src/
    <service>/
      <page>.tsx        # one entry point per page
  dist/                 # build output — gitignored
    <service>/
      <page>.js
```

Source files are organised by owning service under `src/<service>/`. For example, `src/ci/flaky_tests.tsx` is the flaky test dashboard served by the CI service. Each source file is a standalone React app (calls `createRoot`, mounts to a known DOM id) and becomes one output bundle under `dist/<service>/`.

## Build

```bash
cd services/ui
npm ci           # install deps from lockfile (first time or after package changes)
npm run build    # esbuild → dist/
npm run check    # tsc --noEmit — type-check only, no output
```

### How artifacts get into a service

Compiled bundles are not committed. Each service that uses a bundle has a Makefile target that builds it and copies it into the service's static tree. For example:

```makefile
services/ui/dist/ci/flaky_tests.js: $(shell git ls-files services/ui)
	cd services/ui && npm ci && npm run build

ci/ci/static/compiled-js/flaky_tests.js: services/ui/dist/ci/flaky_tests.js
	mkdir -p ci/ci/static/compiled-js
	cp services/ui/dist/ci/flaky_tests.js $@
```

The service image target depends on the copy target, so building the image always picks up the latest compiled JS.

In `build.yaml`, two steps run before any service image that depends on compiled JS is built:
- `build_ui` — runs `npm ci && npm run build`, uploads `dist/` as an artifact.
- `check_ui` — runs `npm ci && npm run check` for type safety.

## Coding conventions

### Component organization

- `services/ui/src/shared/` — genuinely generic components/hooks/utils with no domain-specific coupling (e.g. `Panel`, `SpinnerIcon`, `useLegendToggle`, `timeUtils`). These are candidates for reuse across services.
- `services/ui/src/<service>/components/` — pure functions, types, and components that are cleanly factored but only make sense for one feature or service today (e.g. dollar formatting, stats helpers for a specific dashboard). Don't promote something to `shared/` just because it's generically *coded* — promote it once a second consumer actually needs it. Until then, keeping it service-local signals its actual scope honestly. See `batch/components/jobModels.ts` for a precedent (pure types/utils colocated with a service's components, not in `shared/`).

### `Map` vs `Record` vs a typed interface

- If a value's field set is known and fixed at compile time, use a proper typed interface/class with named properties — not a dynamic collection at all.
- If the field set is dynamic/unknown (e.g. keyed by external data like GCP service/SKU names, or billing project names), prefer `Map<string, T>` over `Record<string, T>`. Bracket access/assignment on a plain object with a computed key is what `eslint-plugin-security`'s `detect-object-injection` rule (wired into Codacy for this repo) flags — a key that ever equals `__proto__`/`constructor` can mutate `Object.prototype` via bracket-assignment on a `Record`, but `Map.set`/`.get` are immune to this regardless of key value.
- **Gotcha when converting `Record` → `Map`:** `Object.entries`/`Object.values`/`Object.keys` on a `Map` instance do **not** raise a TypeScript error (they fall through to a loosely-typed `any`-returning overload) but silently return `[]` at runtime, since a `Map`'s entries aren't stored as the object's own enumerable string-keyed properties. Converting a field from `Record` to `Map` is only safe if you also grep the whole file for `Object.entries(field)` / `Object.values(field)` / `Object.keys(field)` and convert those to `[...field]` / `[...field.values()]` / `[...field.keys()]` (or direct `for...of` iteration — `Map` is natively iterable as `[k, v]` pairs). `tsc --noEmit` reliably catches stray bracket-access sites during this kind of migration, but it will not catch this one — check for it by hand.

### CSP headers — pick the right decorator, in both places

Every route handler is wrapped in one of `web_security_headers` / `web_security_headers_inline_styles` / `web_security_headers_swagger` (from `web_common`), which controls the response's `Content-Security-Policy`. The default (`web_security_headers`) has no `style-src 'unsafe-inline'`, so **any page whose React tree emits an inline `style=` attribute must use `web_security_headers_inline_styles` instead**, or the browser silently drops those styles (broken chart sizing/layout, no error in the console). This bites recharts-heavy pages especially, since recharts computes a lot of its layout via inline styles.

This has to be set in **two** places, and it's easy to fix one and forget the other:
- The real route handler (e.g. `monitoring/monitoring/monitoring.py`).
- The matching entry in `_LOCAL_REACT_ROUTES` in `devbin/dev_proxy.py`, which picks a decorator by template name (see the `_template == 'index_react.html'` check). If your new template isn't in that check, local dev silently gets the wrong CSP and won't reproduce the production bug — you have to add it explicitly.

When adding a new React page, check what decorator the closest existing page of the same shape uses (e.g. `/billing` for anything chart-heavy) rather than defaulting to plain `web_security_headers`.

### Fetching data: use the `hailApi` client objects

When calling an API from a page, use the per-service `hailApi` client objects (e.g. `services/ui/src/monitoring/components/hailApi.ts`, built on `hailApiFetch` from `services/ui/src/shared/hailApiFetch.ts`) — not a raw `fetch()`. They handle the CSRF header, credentials, and JSON parsing/error handling for you, and type both the call and the response.

If the endpoint you need isn't in the client yet, add it — write its signature and response type by reference to that service's `/openapi.yaml`, not by guessing from the Python handler. Only add endpoints a page actually calls; these clients are a mirror of current usage, not the full API surface.

## Local development with the dev-proxy

You don't need a deployed environment to work on a React page. The dev-proxy (`devbin/dev_proxy.py`) runs a local aiohttp server that renders Jinja2 templates directly and either proxies API calls to a real deployed service or serves mock data locally.

### With mock data (no deployed service needed)

For pages that have mock data handlers in the dev-proxy, set `MOCK_API_DATA=1`. For example, to work on the CI flaky tests dashboard:

```bash
MOCK_API_DATA=1 SERVICE=ci make devserver
```

This single command installs npm deps if needed, starts esbuild in watch mode (recompiling `.tsx` on every save), and runs the dev-proxy — all in parallel. Open (eg) `http://localhost:8000/flaky_tests` in the browser. Changes to `.tsx` files are picked up automatically.

`MOCK_API_DATA=1` activates hardcoded fixture data in the proxy, so the page renders fully without touching any real database or deployed cluster. When adding a new page, add a corresponding mock handler to `devbin/dev_proxy.py`.

### Against a real deployed service

Omit `MOCK_API_DATA` and make sure you have valid Hail credentials configured. The proxy will forward API requests to the service at whatever namespace your deploy config points to.
