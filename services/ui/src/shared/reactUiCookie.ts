// The "Back to classic layout" escape hatch these pages offer: clears the opt-in cookie that
// routes the request to the React page, then reloads so the server renders the Jinja2 version.
export function disableReactUi(): void {
  document.cookie = 'hail_react_ui=; max-age=0; path=/; SameSite=Lax';
  location.reload();
}
