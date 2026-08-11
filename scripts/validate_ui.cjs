const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const output = path.join(root, "output", "ui", "screenshots");
fs.mkdirSync(output, { recursive: true });

const pages = [
  ["dashboard", "/"],
  ["top-signals", "/signals"],
  ["universe", "/universe"],
  ["sector-map", "/sectors"],
  ["regional-markets", "/regions"],
  ["etf-explorer", "/etfs"],
  ["commodity-dashboard", "/commodities"],
  ["strategies", "/strategies"],
  ["portfolio", "/portfolio"],
  ["pnl-calendar", "/performance"],
  ["market-news", "/news"],
  ["asset-analysis", "/asset/AAPL"],
  ["system-health", "/health"],
];

(async () => {
  const browser = await chromium.launch({ headless: true });
  const results = [];
  for (const [name, route] of pages) {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      deviceScaleFactor: 1,
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    const response = await page.goto(
      `http://127.0.0.1:8080${route}`,
      { waitUntil: "domcontentloaded", timeout: 30000 },
    );
    await page.waitForTimeout(900);
    const layout = await page.evaluate(() => {
      const sidebar = document.querySelector(".sidebar")?.getBoundingClientRect();
      const main = document.querySelector("main")?.getBoundingClientRect();
      const content = document.querySelector(".content");
      return {
        horizontalOverflow:
          document.documentElement.scrollWidth >
          document.documentElement.clientWidth + 2,
        sidebarOverlap:
          Boolean(sidebar && main) && main.left < sidebar.right - 1,
        contentTextLength: (content?.innerText || "").trim().length,
        viewportWidth: window.innerWidth,
        documentWidth: document.documentElement.scrollWidth,
      };
    });
    if (name === "top-signals") {
      await page.locator("[data-chart-symbol]").first().click();
      await page.waitForSelector("#price-chart .plot-container", {
        timeout: 15000,
      });
    }
    const screenshot = path.join(output, `${name}-desktop.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    results.push({
      name,
      route,
      status: response?.status(),
      errors,
      layout,
      screenshot: path.relative(root, screenshot).replaceAll("\\", "/"),
    });
    await page.close();
  }

  for (const [name, route] of [
    ["dashboard", "/"],
    ["top-signals", "/signals"],
    ["universe", "/universe"],
    ["regional-markets", "/regions"],
    ["asset-analysis", "/asset/AAPL"],
    ["market-news", "/news"],
    ["portfolio", "/portfolio"],
    ["pnl-calendar", "/performance"],
  ]) {
    const page = await browser.newPage({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 1,
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    await page.goto(`http://127.0.0.1:8080${route}`, {
      waitUntil: "domcontentloaded",
      timeout: 30000,
    });
    await page.waitForTimeout(700);
    const layout = await page.evaluate(() => ({
      horizontalOverflow:
        document.documentElement.scrollWidth >
        document.documentElement.clientWidth + 2,
      contentTextLength:
        (document.querySelector(".content")?.innerText || "").trim().length,
      viewportWidth: window.innerWidth,
      documentWidth: document.documentElement.scrollWidth,
    }));
    const screenshot = path.join(output, `${name}-mobile.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    results.push({
      name: `${name}-mobile`,
      route,
      status: 200,
      errors,
      layout,
      screenshot: path.relative(root, screenshot).replaceAll("\\", "/"),
    });
    await page.close();
  }
  await browser.close();

  const failed = results.filter(
    (result) =>
      result.status !== 200 ||
      result.errors.length ||
      result.layout.horizontalOverflow ||
      result.layout.sidebarOverlap ||
      result.layout.contentTextLength < 80,
  );
  const report = {
    schema: "stocks_ui_visual_validation_v1",
    status: failed.length ? "NO_GO" : "GO",
    generated_at: new Date().toISOString(),
    tested_url: "http://127.0.0.1:8080",
    result_count: results.length,
    failure_count: failed.length,
    results,
  };
  fs.writeFileSync(
    path.join(root, "output", "ui", "visual-validation.json"),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  process.exitCode = failed.length ? 1 : 0;
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
