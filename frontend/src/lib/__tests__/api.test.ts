import { describe, it, expect } from "vitest";
import { getBaseUrl } from "../api";

describe("API module", () => {
  describe("getBaseUrl", () => {
    it("returns an empty string for same-origin requests", () => {
      const url = getBaseUrl();
      expect(url).toBe("");
    });

    it("constructs valid API paths when combined with endpoints", () => {
      const base = getBaseUrl();
      const healthPath = `${base}/api/health`;
      expect(healthPath).toBe("/api/health");
    });

    it("constructs valid paths for nested API endpoints", () => {
      const base = getBaseUrl();
      const path = `${base}/api/chat/conversations`;
      expect(path).toBe("/api/chat/conversations");
    });

    it("constructs valid paths for parameterized endpoints", () => {
      const base = getBaseUrl();
      const id = 42;
      const path = `${base}/api/monitors/${id}`;
      expect(path).toBe("/api/monitors/42");
    });
  });
});
