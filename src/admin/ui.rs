use axum::response::Html;

const ADMIN_UI_HTML: &str = include_str!("ui/admin_panel.html");

// La page HTML est publique — l'auth se fait côté JS via X-Admin-Key
// sur chaque appel API (qui eux sont protégés par require_admin! dans handlers/admin.rs).
pub async fn admin_ui_handler() -> Html<&'static str> {
    Html(ADMIN_UI_HTML)
}
