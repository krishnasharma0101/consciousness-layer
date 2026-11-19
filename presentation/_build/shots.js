const { chromium } = require('playwright-core');
(async () => {
  const [,, file, outdir, sel, prefix, scale] = process.argv;
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 }, deviceScaleFactor: +scale || 2 });
  const errs = []; page.on('console', m => { if (m.type()==='error'||m.type()==='warning') errs.push(m.text()); }); page.on('pageerror', e => errs.push(e.message));
  await page.goto('file://' + file); await page.waitForTimeout(2500);
  await page.addStyleTag({content: 'nav{position:static!important}'});
  const els = await page.$$(sel);
  for (const el of els) { const id = await el.getAttribute('id'); if (!id) continue; await el.scrollIntoViewIfNeeded(); await page.waitForTimeout(300); await el.screenshot({ path: `${outdir}/${prefix}${id}.png` }); console.log('shot', id); }
  console.log('ERRORS', JSON.stringify(errs)); await browser.close();
})();
