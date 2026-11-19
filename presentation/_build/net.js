const { chromium } = require('playwright-core');
(async () => { const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  for (const f of process.argv.slice(2)) { const page = await browser.newPage(); const reqs=[]; const errs=[];
    await page.route('**/*', r => { const u=r.request().url(); if (u.startsWith('file:') || u.startsWith('data:')) return r.continue(); reqs.push(u); return r.abort(); });
    page.on('pageerror', e=>errs.push(e.message)); page.on('console', m=>{ if (m.type()==='error') errs.push(m.text()); });
    await page.goto('file://'+f); await page.waitForTimeout(3000); console.log(f.split('/').pop(), 'non-file requests:', reqs.length, 'errors:', errs.length, JSON.stringify(errs).slice(0,300)); await page.close(); }
  await browser.close(); })();
