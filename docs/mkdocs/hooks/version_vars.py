# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Substitutes version placeholder tokens with the pinned upstream vLLM
release, so install instructions scattered across the docs can be bumped
in one place (the `extra.vllm_version` value in mkdocs.yml) instead of
hunting down every hardcoded occurrence.

Supported tokens:
  - `{{vllm_version}}` -> full release, e.g. "0.27.0"
  - `{{vllm_minor}}`   -> major.minor, e.g. "0.27"

Runs on `on_page_content` (the rendered HTML), not `on_page_markdown`.
Several install docs (e.g. cuda.inc.md, rocm.inc.md) are only ever pulled
into a real page through a pymdownx.snippets `--8<--` include, and that
expansion happens during Markdown parsing, after `on_page_markdown` hooks
have already run on the including page's raw text. By the time
`on_page_content` fires, snippets are expanded, so a plain string
replacement still finds the tokens (`{{...}}` isn't special Markdown
syntax, so it survives rendering unchanged).

`on_post_build` then scans the built site for any token that survived
substitution (e.g. from a typo, a whitespace variant, or a future mkdocs/
pymdownx change that breaks the timing assumption above) and fails the
build loudly, so a broken pin ships as a build error instead of silently
as literal "{{vllm_version}}" text on the published site.
"""

import re
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

_TOKEN_RE = re.compile(r"\{\{\s*vllm_(?:version|minor)\s*\}\}")


def on_page_content(html: str, *, page: Page, config: MkDocsConfig, files: Files) -> str:
    version = config.extra.get("vllm_version")
    if not version:
        return html

    minor = ".".join(version.split(".")[:2])
    html = html.replace("{{vllm_minor}}", minor)
    html = html.replace("{{vllm_version}}", version)
    return html


def on_post_build(*, config: MkDocsConfig) -> None:
    site_dir = Path(config.site_dir)
    leaked = sorted(
        str(path.relative_to(site_dir))
        for path in site_dir.rglob("*.html")
        if _TOKEN_RE.search(path.read_text(encoding="utf-8"))
    )
    if leaked:
        raise PluginError(
            "Unresolved {{vllm_version}}/{{vllm_minor}} token(s) found in built pages "
            f"(hooks/version_vars.py failed to substitute them): {', '.join(leaked)}"
        )
