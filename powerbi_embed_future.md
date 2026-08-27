# Future Upgrade: Live Power BI Embedding

This project currently displays the Power BI dashboard as **static PNG
exports** of each report page (`static/dashboard/powerbi_page-*.png`),
generated once from `data/dashboard.pbix`. This document explains how that
could be replaced with a **live, interactive embedded report** if Power BI
embedding credentials or an activated organizational account become
available later. None of the steps below are required for the app to work
today — they are purely for a future upgrade.

## Why static images are used today

Live Power BI embedding requires one of:

1. **Power BI "Publish to Web"** — needs a Microsoft work/school
   (organizational) account; personal/consumer Microsoft accounts cannot
   use it, and it publishes the report **publicly** on the internet, which
   is not acceptable for many projects.
2. **Power BI Embedded (for your organization)** — needs a Power BI Pro or
   Premium Per User license on the account that owns the report.
3. **Power BI Embedded (embed for your customers)** — needs an Azure
   subscription, an Azure AD app registration, and a Power BI Embedded
   capacity (paid Azure resource).

None of these are available without a university/organizational email or a
paid Azure/Power BI subscription, so the app is built to work fully
**without** any of them.

## Option A — Publish to Web (simplest, if an org account becomes available)

1. Open `data/dashboard.pbix` in Power BI Desktop, signed in with an
   organizational account.
2. **File → Publish → Publish to Power BI** to upload it to a workspace.
3. In the Power BI Service, open the report → **File → Embed report →
   Publish to web (public)**.
4. Copy the generated `<iframe>` embed URL.
5. In `templates/powerbi.html`, replace the `<img>` tags inside
   `.dashboard-page` with:

   ```html
   <iframe
     title="Where Heart Data Becomes Insight"
     width="100%" height="700"
     src="PASTE_PUBLISH_TO_WEB_URL_HERE"
     frameborder="0" allowFullScreen="true">
   </iframe>
   ```

   Note: this makes the report **publicly viewable by anyone with the
   link** — do not use this option if the data must stay private.

## Option B — Power BI Embedded with Azure AD (private, production-grade)

1. Register an app in **Azure Active Directory** and grant it the
   `Report.Read.All` (or narrower) Power BI API permission.
2. Provision a **Power BI Embedded capacity** (or use an existing
   Premium/Fabric capacity) in Azure.
3. Upload `data/dashboard.pbix` to a Power BI workspace assigned to that
   capacity.
4. On the Flask backend, use the `msal` Python package to acquire an Azure
   AD token for the service principal, then call the Power BI REST API
   (`GET /v1.0/myorg/groups/{groupId}/reports/{reportId}`) to obtain an
   **embed URL** and generate an **embed token**
   (`POST /GenerateToken`).
5. Add a new Flask route, e.g. `/api/powerbi-embed-token`, that returns the
   `embedUrl`, `accessToken`, and `reportId` as JSON.
6. On the front end, include the official **`powerbi-client`** JavaScript
   library (via CDN) and use it in `templates/powerbi.html` to render a
   live, interactive report into a `<div id="powerbi-container">`, calling
   `powerbi.embed(container, config)` with the values fetched from the new
   API route.
7. Keep the static PNG fallback in place (e.g. behind a feature flag /
   environment variable such as `POWERBI_EMBED_ENABLED`) so the app still
   works offline or if the Azure credentials expire.

## What does NOT need to change

- `data/dashboard.pbix` stays exactly as it is — no modification needed for
  either embedding option.
- The "Download Original Power BI Dashboard" button and route
  (`/download/pbix` in `app.py`) can remain exactly as-is.
- The rest of the application (prediction, analytics, model performance)
  is entirely independent of how the Power BI page is rendered.
