const { chromium } = require('playwright-core');
(async () => {
  const [,, file, outdir] = process.argv;
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const errs = []; page.on('console', m => { if (m.type()==='error'||m.type()==='warning') errs.push(m.text()); }); page.on('pageerror', e => errs.push(e.message));
  await page.goto('file://' + file); await page.waitForTimeout(3000);
  const clk = async (id, wait) => { await page.click('#' + id); await page.waitForTimeout(wait || 500); };
  await page.screenshot({ path: `${outdir}/graph3d_01_query_t0.png` });
  await clk('run', 9500);                     // run activation to convergence (animated)
  await page.screenshot({ path: `${outdir}/graph3d_02_activation_converged.png` });
  await clk('astar', 2500);
  await page.screenshot({ path: `${outdir}/graph3d_03_astar_path.png` });
  await clk('clearPath', 300); await clk('fit', 1200);
  await clk('walk', 450 * 31 + 800);
  await page.screenshot({ path: `${outdir}/graph3d_04_random_walk.png` });
  await clk('clearPath', 300);
  await page.check('input[name=lay][value=free]'); await page.waitForTimeout(1500); await clk('run', 9500);
  await page.screenshot({ path: `${outdir}/graph3d_05_free_layout.png` });
  await page.check('input[name=lay][value=time]'); await page.waitForTimeout(1200);
  await page.click('#shuf'); await page.waitForTimeout(500); await clk('run', 9500);
  await page.screenshot({ path: `${outdir}/graph3d_06_shuffled_W_control.png` });
  console.log('ERRORS', JSON.stringify(errs)); await browser.close();
})();
