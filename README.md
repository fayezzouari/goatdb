# site-assets

Large media for the goatdb website, kept off `master` so it never ships in
the Go module, release archives or the Docker image.

The Pages workflow (`.github/workflows/pages.yml` on `master`) copies these
files into `site/public/` at build time. For local development run
`npm run fetch-video` in `site/`.

- `goatdb-demo.mp4`: the demo video. Soundtrack: "One Cool Minute" by
  Loyalty Freak Music, CC0.
