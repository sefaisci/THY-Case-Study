"""Code-native design tokens extracted from the accepted THY UI concept."""

THY_CSS = """
<style>
:root {
  --thy-red: #d71920;
  --thy-red-dark: #b81016;
  --thy-navy: #17243a;
  --thy-muted: #607089;
  --thy-border: #dce2ea;
  --thy-surface: #f7f9fc;
  --thy-white: #ffffff;
}

html, body, [class*="css"] {
  font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--thy-navy);
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--thy-white);
}

[data-testid="stHeader"], [data-testid="stToolbar"], footer {
  display: none;
}

.block-container {
  max-width: 100% !important;
  padding: 1rem 1.1rem 1.4rem !important;
}

h1, h2, h3, h4, label {
  color: var(--thy-navy);
}

h1 { font-size: 1.9rem !important; letter-spacing: -0.025em; }
h2 { font-size: 1.22rem !important; letter-spacing: -0.012em; }
h3 { font-size: 1rem !important; }

[data-testid="stHorizontalBlock"] { gap: 1.25rem; }

div[data-testid="stColumn"]:has(.thy-left-marker) {
  border-right: 1px solid var(--thy-border);
  padding-right: 1.1rem;
}

div[data-testid="stColumn"]:has(.thy-right-marker) {
  border-left: 1px solid var(--thy-border);
  padding-left: 1.1rem;
}

.thy-column-marker { display: none; }

div[data-testid="stColumn"]:has(.thy-left-marker)
div[data-testid="stFullScreenFrame"]:has(div[data-testid="stImage"]) {
  display: flex;
  justify-content: center;
}

div[data-testid="stColumn"]:has(.thy-left-marker) div[data-testid="stImage"] {
  display: flex;
  justify-content: center;
  margin: .15rem 0 .45rem;
}

div[data-testid="stColumn"]:has(.thy-left-marker) div[data-testid="stImage"] img {
  width: 64px !important;
  height: auto !important;
  object-fit: contain;
}

div[data-testid="stColumn"]:has(.thy-left-marker)
div[data-testid="stElementContainer"]:has(.thy-brand-title) {
  margin-top: -.7rem;
}

.thy-brand-title {
  font-size: 1.05rem;
  line-height: 1.2;
  font-weight: 750;
  color: var(--thy-navy);
  margin: 0 0 1.1rem;
  padding: 0;
  text-align: center;
  white-space: nowrap;
}

.thy-section-rule {
  height: 1px;
  background: var(--thy-border);
  margin: .2rem 0 1rem;
}

.thy-empty {
  border: 1px dashed #b8c2d0;
  border-radius: 10px;
  padding: 2rem 1rem;
  text-align: center;
  color: var(--thy-muted);
  background: var(--thy-white);
}

.thy-citation {
  border-top: 1px solid var(--thy-border);
  padding: .72rem 0 .62rem;
}

.thy-citation strong { font-size: .88rem; color: var(--thy-navy); }
.thy-citation-meta { color: var(--thy-muted); font-size: .76rem; margin: .16rem 0 .35rem; }
.thy-citation-excerpt { color: #36445a; font-size: .82rem; line-height: 1.45; }

.thy-document-row {
  border-top: 1px solid var(--thy-border);
  padding: .6rem 0 .25rem;
}
.thy-document-name { font-weight: 700; font-size: .85rem; overflow-wrap: anywhere; }
.thy-document-meta { color: var(--thy-muted); font-size: .74rem; line-height: 1.45; }

.thy-status {
  display: inline-block;
  width: .52rem;
  height: .52rem;
  border-radius: 50%;
  margin-right: .34rem;
  background: #8793a5;
}
.thy-status.completed { background: #138a45; }
.thy-status.processing { background: #246fce; }
.thy-status.pending { background: #d88900; }
.thy-status.failed, .thy-status.deletion_pending { background: var(--thy-red); }

button[data-testid^="stBaseButton"] {
  border-radius: 7px !important;
  min-height: 2.45rem;
  font-weight: 650;
}
button[data-testid="stBaseButton-primary"] {
  background: var(--thy-red) !important;
  border-color: var(--thy-red) !important;
  color: white !important;
}
button[data-testid="stBaseButton-primary"]:hover {
  background: var(--thy-red-dark) !important;
  border-color: var(--thy-red-dark) !important;
}

button[data-testid^="stBaseButton"]:disabled {
  background: #e9edf3 !important;
  border-color: #d4dae3 !important;
  color: #68758a !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
}

input:focus, textarea:focus, [data-baseweb="select"]:focus-within {
  border-color: var(--thy-red) !important;
  box-shadow: 0 0 0 1px var(--thy-red) !important;
}

button:focus-visible, [role="button"]:focus-visible {
  outline: 3px solid #235ea7 !important;
  outline-offset: 2px !important;
}

[data-testid="stChatMessage"] {
  background: transparent;
  border-bottom: 1px solid #edf0f4;
  padding: .85rem .2rem 1rem;
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--thy-white);
  border: 1px dashed #aeb9c9;
  border-radius: 10px;
}

[data-testid="stExpander"] {
  border: 1px solid var(--thy-border) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}

@media (max-width: 1050px) {
  .block-container { padding: .75rem !important; }
  [data-testid="stHorizontalBlock"] { gap: .65rem; }
  h1 { font-size: 1.5rem !important; }
}
</style>
"""
