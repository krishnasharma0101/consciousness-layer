const { chromium } = require('playwright-core');
(async () => {
  const [,, file, png, w, h, extra] = process.argv;
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: +w || 1600, height: +h || 900 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on('console', m => { if (m.type() === 'error' || m.type()==='warning') errs.push(m.type()+': '+m.text()); });
  page.on('pageerror', e => errs.push('pageerror: ' + e.message));
  page.on('requestfailed', r => errs.push('requestfailed: ' + r.url()));
  const t0 = Date.now();
  await page.goto('file://' + file, { waitUntil: 'load' });
  await page.waitForTimeout(2500);
  if (extra) { const r = await page.evaluate(extra); console.log('EVAL', JSON.stringify(r, null, 0)); await page.waitForTimeout(2500); }
  await page.screenshot({ path: png });
  console.log('load+wait ms', Date.now() - t0);
  console.log('ERRORS', JSON.stringify(errs));
  await browser.close();
})();
