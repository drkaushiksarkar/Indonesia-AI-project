# GitHub and Output-Sharing Runbook

## Recommended release architecture

Use three distribution surfaces:

1. A private GitHub repository for source code, SQL, configuration, tests and continuous integration.
2. GitHub Pages for the curated, non-sensitive HTML reports and image gallery only after public-release approval.
3. An access-controlled object store or managed file-sharing service for the complete delivery archive and PostgreSQL dump. Use Zenodo only when public redistribution and licensing are approved and a DOI is desirable.

Do not commit raw downloads, PostgreSQL dumps, generated tables, credentials or the full client archive to Git history.

## Create and push the code repository

Authenticate with GitHub CLI, then run the following commands from this directory after replacing `OWNER` and `REPOSITORY`.

```bash
gh auth login
git init -b main
git add .
git diff --cached --check
git commit -m "Initial reproducible release"
gh repo create OWNER/REPOSITORY --private --source=. --remote=origin --push
```

Keep the repository private until the source-license review, secret scan, output privacy review and public-data redistribution review are complete.

## Publish the analytical site

The `docs/` directory contains a static landing page, the validated predictability report and curated figures. To publish it with GitHub Pages, open the repository settings, choose Pages, select deployment from a branch, and set `main` with `/docs` as the source.

GitHub Pages should be treated as public internet publishing. For confidential circulation, zip `docs/` and share it through an access-controlled drive or object-store link instead.

## Share the database and complete delivery

Do not distribute the 2026-08-30 dated delivery or its outer archive. Audit sampling found 3,691 zero-length members inside the project archive, consistent with iCloud placeholder files being captured before hydration. The live PostgreSQL lake is intact, but the complete delivery must be rebuilt from fully hydrated source files and a fresh database dump before release.

GitHub blocks ordinary repository files above 100 MiB. Individual GitHub Release assets may be below 2 GiB, but GitHub advises using a file-sharing service for large databases.

Preferred sequence:

1. Hydrate every project file locally and verify that expected files are non-empty.
2. Create a fresh PostgreSQL dump from the live lake and validate a restore into a temporary database.
3. Build a new complete delivery archive, record its manifest and SHA-256 checksum, and sample-extract every file class.
4. Upload only the rebuilt, validated delivery to the chosen access-controlled store.
5. Share a read-only link with an expiry date when the platform supports it.
6. Publish a GitHub Release containing code and lightweight documentation.
7. If public scientific preservation is approved, connect the GitHub repository to Zenodo and archive a tagged release to obtain a DOI.

## Release gates

- Confirm repository owner and visibility.
- Select a software license.
- Confirm redistribution rights for each public source.
- Re-run lint, tests, secret scanning and personal-path scanning.
- Verify the HTML reports and every image in the final hosting environment.
- Record release tag, database dump checksum, project archive checksum and data snapshot date.

## Official references

- GitHub large-file guidance: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub release assets: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- GitHub Pages setup: https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
- GitHub CLI repository creation: https://cli.github.com/manual/gh_repo_create
- Zenodo GitHub integration: https://help.zenodo.org/docs/github/
