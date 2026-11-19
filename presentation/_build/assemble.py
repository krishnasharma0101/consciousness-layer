import json, os, sys
B = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(B); NM = "/workspace/pres-build/node_modules"
def lib(p): return open(os.path.join(NM, p)).read().replace("</script", "<\\/script")
def data(obj): return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
def build(tpl, out, libs, d):
    s = open(os.path.join(B, tpl)).read()
    s = s.replace("/*__LIB__*/", "\n".join(lib(x) for x in libs)).replace("/*__DATA__*/", data(d))
    open(os.path.join(P, out), "w").write(s); print(out, os.path.getsize(os.path.join(P, out)))
which = sys.argv[1:] or ["graph", "results", "index"]
if "graph" in which:
    build("graph3d.tpl.html", "graph3d.html", ["3d-force-graph/dist/3d-force-graph.min.js"], json.load(open(os.path.join(B, "graph_data.json"))))
if "results" in which:
    build("results.tpl.html", "results.html", ["plotly.js-dist-min/plotly.min.js"], json.load(open(os.path.join(B, "results_data.json"))))
if "index" in which:
    import base64, io
    from PIL import Image
    def thumb(path, w=900):
        im = Image.open(path).convert("RGB"); h = int(im.height * w / im.width); im = im.resize((w, h), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "JPEG", quality=82); return base64.b64encode(b.getvalue()).decode()
    s = open(os.path.join(B, "index.tpl.html")).read()
    s = s.replace("/*__THUMB_GRAPH__*/", thumb(os.path.join(P, "png", "graph3d_02_activation_converged.png")))
    s = s.replace("/*__THUMB_RESULTS__*/", thumb(os.path.join(P, "png", "results_section_s6.png")))
    open(os.path.join(P, "index.html"), "w").write(s); print("index.html", os.path.getsize(os.path.join(P, "index.html")))
