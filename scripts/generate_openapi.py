"""Script to pre-generate OpenAPI schema and ReDoc HTML for static documentation hosting."""

from __future__ import annotations

import json
from pathlib import Path
from app.main import app

def main() -> None:
    # Ensure target directories exist
    api_dir = Path("docs/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    # Generate openapi.json
    openapi_data = app.openapi()
    
    # Save the OpenAPI schema
    openapi_file = api_dir / "openapi.json"
    openapi_file.write_text(json.dumps(openapi_data, indent=2))
    print(f"Generated {openapi_file}")

    # Generate standalone ReDoc index.html
    redoc_html = """<!DOCTYPE html>
<html>
  <head>
    <title>ExpYT API Reference</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
      body {
        margin: 0;
        padding: 0;
        background-color: #fafafa;
      }
    </style>
  </head>
  <body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0-rc.76/bundles/redoc.standalone.js"></script>
    <script>
      Redoc.init(
        'openapi.json',
        {
          theme: {
            colors: {
              primary: { main: '#3F51B5' }
            }
          }
        },
        document.getElementById('redoc-container')
      )
    </script>
  </body>
</html>
"""
    html_file = api_dir / "index.html"
    html_file.write_text(redoc_html)
    print(f"Generated {html_file}")

if __name__ == "__main__":
    main()
