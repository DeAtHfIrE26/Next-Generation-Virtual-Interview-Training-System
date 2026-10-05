import { afterEach, describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { proxy } from "../proxy";

const req = (auth?: string) =>
  new NextRequest("https://example.test/dashboard", { headers: auth ? { authorization: auth } : {} });

describe("site password gate", () => {
  afterEach(() => {
    delete process.env.SITE_PASSWORD;
  });

  it("is open when SITE_PASSWORD is unset", () => {
    expect(proxy(req()).status).toBe(200);
  });

  it("challenges without or with wrong credentials, admits the right password", () => {
    process.env.SITE_PASSWORD = "dummy-pass";
    expect(proxy(req()).status).toBe(401);
    expect(proxy(req()).headers.get("www-authenticate")).toContain("Basic");
    expect(proxy(req(`Basic ${btoa("x:wrong")}`)).status).toBe(401);
    expect(proxy(req("Basic %%%")).status).toBe(401);
    expect(proxy(req(`Basic ${btoa("anyone:dummy-pass")}`)).status).toBe(200);
  });
});
