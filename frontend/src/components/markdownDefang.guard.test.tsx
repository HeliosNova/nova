// @ts-nocheck — node-context meta-test: scans the source tree with node:fs.
// The browser app's tsconfig has no @types/node, so tsc can't type these
// built-ins; vitest (esbuild) runs it fine. Type-checking a filesystem scan
// adds no value, so this one file opts out rather than adding a dep.
import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

// Security invariant (audit 2026-08-23). Every <ReactMarkdown> that renders
// untrusted content MUST pass components={markdownComponents}, whose `img`
// override defangs external images to non-fetching links. Monitor digests and
// dossiers are built from ingested web content, so a BARE ReactMarkdown lets
// an injected image tag auto-fetch on render = silent exfiltration/tracking.
// The chat surface was defanged 2026-08-22; this guard makes the invariant
// hold across ALL current and future markdown surfaces, not just chat.
//
// SRC_DIR is the app source tree (vitest runs from the frontend/ project root).
// Test/spec files are skipped (not shipped, and they legitimately contain the
// literal string in comments/regex).
const SRC_DIR = join(process.cwd(), "src");

function walkTsx(dir: string): string[] {
  const out: string[] = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) {
      if (e.name === "node_modules" || e.name === "dist") continue;
      out.push(...walkTsx(join(dir, e.name)));
    } else if (/\.tsx?$/.test(e.name) && !/\.(test|spec)\.tsx?$/.test(e.name)) {
      out.push(join(dir, e.name));
    }
  }
  return out;
}

describe("markdown defang invariant", () => {
  it("no <ReactMarkdown> renders without components={markdownComponents}", () => {
    const offenders: string[] = [];
    for (const file of walkTsx(SRC_DIR)) {
      const src = readFileSync(file, "utf8");
      const tags = src.match(/<ReactMarkdown\b[^>]*>/g) || [];
      for (const tag of tags) {
        if (!/components=/.test(tag)) {
          offenders.push(`${file}: ${tag.replace(/\s+/g, " ").slice(0, 70)}`);
        }
      }
    }
    expect(
      offenders,
      `Bare <ReactMarkdown> (external-image exfil risk) — add components={markdownComponents}:\n${offenders.join("\n")}`,
    ).toEqual([]);
  });
});
