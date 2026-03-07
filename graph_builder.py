"""
graph_builder.py — Domain-aware knowledge graph extraction.

Improvement: extraction prompt now includes the document domain so the LLM
produces domain-relevant entity categories and relationships rather than
generic noun-chunk pairs. A DevOps doc gets nodes like Container/Orchestration/
Service, not generic "Process/Concept/Other".
"""
import os
import re
import json
from collections import defaultdict, Counter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from document_loader import smart_load
from cache_manager import get_graph_cache

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

CHUNK_CHARS = 2500   # larger chunks → more context per extraction call
MAX_CHUNKS  = 12
TOP_NODES   = 35
TOP_EDGES   = 60

CATEGORY_COLORS = {
    "technology":   "#4FC3F7",
    "organization": "#81C784",
    "person":       "#FFB74D",
    "place":        "#FF8A65",
    "concept":      "#80DEEA",
    "process":      "#CE93D8",
    "product":      "#F48FB1",
    "benefit":      "#A5D6A7",
    "challenge":    "#EF9A9A",
    "other":        "#B0BEC5",
}
VALID_CATS = set(CATEGORY_COLORS.keys())

SKIP_NODES = {
    "it","this","they","he","she","we","you","i","document","text",
    "page","article","section","figure","table","example","note",
    "introduction","conclusion","abstract","overview","summary"
}


def _graph_llm(prompt: str) -> str:
    cache = get_graph_cache()
    hit   = cache.get(prompt)
    if hit:
        return hit
    result = llm.invoke(prompt).content.strip()
    cache.set(prompt, result)
    return result


def load_and_chunk(file_path: str) -> list[str]:
    try:
        docs = smart_load(file_path)
    except Exception as e:
        print(f"[Graph] Load error: {e}")
        return []
    if not docs:
        return []
    full_text = re.sub(r'\s+', ' ', "\n\n".join(d.page_content for d in docs)).strip()
    chunks = [
        full_text[i: i + CHUNK_CHARS].strip()
        for i in range(0, len(full_text), CHUNK_CHARS)
        if len(full_text[i: i + CHUNK_CHARS].strip()) > 80
    ]
    return chunks[:MAX_CHUNKS]


def _infer_topic_and_domain(chunk: str) -> tuple[str, str]:
    """Returns (topic_phrase, domain_tag) from the first chunk."""
    raw = _graph_llm(
        f"For this text, provide:\n"
        f"1. Topic: a 5-8 word phrase describing the main subject\n"
        f"2. Domain: 1-3 words for the field (e.g. software engineering, urban farming, finance)\n\n"
        f"Text:\n{chunk[:800]}\n\n"
        f"Reply in exactly this format:\n"
        f"Topic: <topic>\n"
        f"Domain: <domain>"
    )
    topic  = "the document"
    domain = "general"
    for line in raw.splitlines():
        if line.lower().startswith("topic:"):
            topic  = line.split(":",1)[1].strip()
        elif line.lower().startswith("domain:"):
            domain = line.split(":",1)[1].strip().lower()
    return topic, domain


def _extract_triples(chunk: str, topic: str, domain: str) -> list[dict]:
    """
    Domain-aware triple extraction.
    The prompt gives the LLM context about what kind of entities to look for,
    so a software doc produces Container/Service/Orchestrator nodes, not
    generic Concept/Other nodes.
    """
    prompt = f"""You are building a knowledge graph for a {domain} document about: {topic}

Extract the most important factual relationships from the text below.

RULES:
- "s" (subject) and "o" (object): 1-5 words, Title Case
  - For {domain}: focus on key technical terms, systems, components, tools, methods
  - NO generic words like: it, this, they, document, text, page, section
  - NO articles (a, an, the) at the start
- "r" (relationship): 2-5 word verb phrase, lowercase
  - Be specific: not just "uses" but "is orchestrated by", "depends on", "exposes API for"
- "cat_s" / "cat_o": pick the MOST SPECIFIC category that fits in {domain} context:
  technology / organization / person / place / concept / process / product / benefit / challenge / other
- Extract 8-15 triples — include ALL important relationships, not just obvious ones
- SKIP: generic filler, authors, citations, page references

Text:
{chunk}

Return ONLY a valid JSON array (no markdown, no explanation):
[{{"s":"...", "r":"...", "o":"...", "cat_s":"...", "cat_o":"..."}}]"""

    raw = _graph_llm(prompt)
    try:
        raw   = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return []
        triples = json.loads(match.group())
        valid   = []
        for t in triples:
            s  = t.get("s", "").strip().title()
            r  = t.get("r", "").strip().lower()
            o  = t.get("o", "").strip().title()
            cs = t.get("cat_s", "other").lower()
            co = t.get("cat_o", "other").lower()
            if not s or not r or not o:
                continue
            if s.lower() == o.lower():
                continue
            if len(s) < 2 or len(o) < 2 or len(s) > 50 or len(o) > 50:
                continue
            if s.lower() in SKIP_NODES or o.lower() in SKIP_NODES:
                continue
            # Strip leading articles that slipped through
            for art in ("A ", "An ", "The "):
                if s.startswith(art): s = s[len(art):]
                if o.startswith(art): o = o[len(art):]
            valid.append({
                "s": s, "r": r, "o": o,
                "cat_s": cs if cs in VALID_CATS else "other",
                "cat_o": co if co in VALID_CATS else "other",
            })
        return valid
    except Exception as e:
        print(f"[Triple Error] {e}")
        return []


def extract_all_triples(chunks: list[str]) -> tuple[list[dict], str]:
    """Returns (all_triples, domain)."""
    topic, domain = _infer_topic_and_domain(chunks[0]) if chunks else ("document", "general")
    print(f"[Graph] Topic: '{topic}' | Domain: '{domain}'")

    all_triples = []
    for i, chunk in enumerate(chunks):
        print(f"[Graph] Chunk {i+1}/{len(chunks)}…")
        all_triples.extend(_extract_triples(chunk, topic, domain))

    print(f"[Graph] Raw triples: {len(all_triples)}")
    return all_triples, domain


def _stem(name: str) -> str:
    n = name.lower().strip()
    for suf in ("ing", "tion", "ation", "er", "ers", "s"):
        if n.endswith(suf) and len(n) - len(suf) >= 4:
            return n[:-len(suf)]
    return n


def build_graph_data(triples: list[dict]) -> tuple[dict, list]:
    raw_cats: dict[str, list] = defaultdict(list)
    raw_freq: dict[str, int]  = defaultdict(int)

    for t in triples:
        raw_cats[t["s"]].append(t["cat_s"])
        raw_cats[t["o"]].append(t["cat_o"])
        raw_freq[t["s"]] += 1
        raw_freq[t["o"]] += 1

    # Normalise near-duplicate names via stemming
    stem_groups: dict[str, list] = defaultdict(list)
    for name in raw_cats:
        stem_groups[_stem(name)].append(name)

    canonical: dict[str, str] = {}
    for names in stem_groups.values():
        canon = max(names, key=lambda n: raw_freq[n])
        for n in names:
            canonical[n] = canon

    node_meta: dict[str, dict] = defaultdict(lambda: {"category": "other", "freq": 0, "votes": []})
    for name, cats in raw_cats.items():
        c = canonical[name]
        node_meta[c]["freq"]  += raw_freq[name]
        node_meta[c]["votes"] += cats

    for meta in node_meta.values():
        v = meta.pop("votes")
        meta["category"] = Counter(v).most_common(1)[0][0] if v else "other"

    top_set = set(
        sorted(node_meta, key=lambda n: node_meta[n]["freq"], reverse=True)[:TOP_NODES]
    )

    edge_acc: dict[tuple, dict] = defaultdict(lambda: {"labels": [], "weight": 0})
    for t in triples:
        s = canonical.get(t["s"], t["s"])
        o = canonical.get(t["o"], t["o"])
        if s not in top_set or o not in top_set or s == o:
            continue
        edge_acc[(s, o)]["weight"] += 1
        edge_acc[(s, o)]["labels"].append(t["r"])

    edges = []
    for (s, o), data in sorted(edge_acc.items(), key=lambda x: x[1]["weight"], reverse=True):
        label = Counter(data["labels"]).most_common(1)[0][0]
        edges.append({"from": s, "to": o, "label": label, "weight": data["weight"]})
        if len(edges) >= TOP_EDGES:
            break

    used        = {e["from"] for e in edges} | {e["to"] for e in edges}
    final_nodes = {n: node_meta[n] for n in used}
    print(f"[Graph] Final: {len(final_nodes)} nodes, {len(edges)} edges")
    return final_nodes, edges


def render_graph(node_meta: dict, edges: list, domain: str, output_path: str) -> str:
    nodes_js, edges_js = [], []

    for name, meta in node_meta.items():
        cat   = meta.get("category", "other")
        freq  = meta.get("freq", 1)
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        size  = max(18, min(55, 14 + freq * 4))
        label = name if len(name) <= 22 else name[:20] + "…"
        nodes_js.append({
            "id": name, "label": label,
            "title": f"<b>{name}</b><br>Type: {cat}<br>Mentions: {freq}",
            "color": {"background": color, "border": "#111",
                      "highlight": {"background": "#FFD700", "border": "#FF8C00"},
                      "hover":     {"background": "#FFD700", "border": "#FF8C00"}},
            "size": size,
            "font": {"color": "#f0f0f0", "size": 13, "face": "Segoe UI,sans-serif"},
            "borderWidth": 2,
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.7)", "size": 12},
        })

    for e in edges:
        w = e["weight"]
        edges_js.append({
            "from": e["from"], "to": e["to"], "label": e["label"],
            "title": f"{e['from']}  →  {e['label']}  →  {e['to']}",
            "width": max(1, min(7, 1 + w)),
            "color": {"color": "#3a3a6a", "highlight": "#FFD700", "hover": "#90CAF9", "opacity": 0.88},
            "font":  {"color": "#90CAF9", "size": 11, "align": "middle",
                      "strokeWidth": 2, "strokeColor": "#050510"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
        })

    nj = json.dumps(nodes_js, ensure_ascii=False)
    ej = json.dumps(edges_js, ensure_ascii=False)

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:3px 10px;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{c};margin-right:5px;"></span>'
        f'<span style="color:#bbb;font-size:11px;text-transform:capitalize;">{cat}</span></span>'
        for cat, c in CATEGORY_COLORS.items() if cat != "other"
    )

    html = f"""<!DOCTYPE html><html>
<head><meta charset="utf-8"/><title>Knowledge Graph — {domain}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"/>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#09090f;font-family:'Segoe UI',sans-serif;color:#f0f0f0;height:100vh;display:flex;flex-direction:column;overflow:hidden;}}
#topbar{{padding:9px 16px;background:linear-gradient(135deg,#12122a,#1a1a3e);border-bottom:1px solid #2a2a4a;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}}
#topbar h2{{font-size:14px;color:#80DEEA;letter-spacing:1.5px;}}
#meta{{font-size:11px;color:#555;}}
#legend{{padding:4px 16px;background:#0c0c1e;border-bottom:1px solid #1a1a3a;flex-shrink:0;display:flex;flex-wrap:wrap;align-items:center;}}
#wrap{{flex:1;position:relative;overflow:hidden;}}
#graph{{width:100%;height:100%;background:radial-gradient(ellipse at 50% 35%,#0d0d28 0%,#050510 100%);}}
#controls{{position:absolute;top:10px;right:10px;display:flex;flex-direction:column;gap:4px;}}
.btn{{background:#131330;border:1px solid #2a2a5a;color:#80DEEA;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;transition:.15s;white-space:nowrap;}}
.btn:hover{{background:#1e1e4a;border-color:#80DEEA;}}
#searchbox{{position:absolute;top:10px;left:10px;}}
#searchbox input{{background:#131330;border:1px solid #2a2a5a;color:#f0f0f0;padding:5px 10px;border-radius:4px;font-size:12px;width:200px;outline:none;}}
#searchbox input:focus{{border-color:#80DEEA;}}
#tip{{position:absolute;bottom:10px;left:10px;background:#131330dd;border:1px solid #2a2a4a;border-radius:6px;padding:8px 12px;font-size:12px;color:#ccc;max-width:300px;pointer-events:none;display:none;line-height:1.5;}}
</style></head>
<body>
<div id="topbar">
  <h2>⬡ &nbsp;Knowledge Graph</h2>
  <span id="meta">{len(nodes_js)} nodes · {len(edges_js)} edges · domain: {domain}</span>
</div>
<div id="legend">{legend}</div>
<div id="wrap">
  <div id="graph"></div>
  <div id="searchbox"><input type="text" placeholder="🔍 Search…" oninput="searchNode(this.value)"/></div>
  <div id="controls">
    <button class="btn" onclick="net.fit({{animation:true}})">⊡ Fit All</button>
    <button class="btn" onclick="togglePhysics()">⚙ Physics</button>
    <button class="btn" onclick="resetHL()">↺ Reset</button>
  </div>
  <div id="tip"></div>
</div>
<script>
const nodes=new vis.DataSet({nj});
const edges=new vis.DataSet({ej});
const net=new vis.Network(document.getElementById('graph'),{{nodes,edges}},{{
  physics:{{enabled:true,forceAtlas2Based:{{gravitationalConstant:-100,centralGravity:0.005,
    springLength:220,springConstant:0.035,damping:0.6,avoidOverlap:1.3}},
    solver:'forceAtlas2Based',stabilization:{{iterations:300,updateInterval:20}},minVelocity:0.5}},
  interaction:{{hover:true,tooltipDelay:60,hideEdgesOnDrag:true,keyboard:{{enabled:true}}}},
  nodes:{{shape:'dot',shadow:true}},
  edges:{{smooth:{{type:'curvedCW',roundness:0.15}}}},
}});
let physOn=true;
net.on('stabilizationIterationsDone',()=>{{net.setOptions({{physics:{{enabled:false}}}});physOn=false;net.fit({{animation:{{duration:800}}}});}});
function togglePhysics(){{physOn=!physOn;net.setOptions({{physics:{{enabled:physOn}}}});}}
function resetHL(){{nodes.update(nodes.getIds().map(id=>({{id,opacity:1}})));edges.update(edges.getIds().map(id=>({{id,color:{{opacity:0.88}}}})));net.fit({{animation:true}});}}
const tip=document.getElementById('tip');
net.on('hoverNode',p=>{{const n=nodes.get(p.node);tip.innerHTML=n.title;tip.style.display='block';}});
net.on('blurNode',()=>tip.style.display='none');
net.on('hoverEdge',p=>{{const e=edges.get(p.edge);tip.innerHTML=e.title||'';tip.style.display='block';}});
net.on('blurEdge',()=>tip.style.display='none');
net.on('click',p=>{{
  if(p.nodes.length>0){{
    const nid=p.nodes[0];const conn=new Set(net.getConnectedNodes(nid));conn.add(nid);
    nodes.update(nodes.getIds().map(id=>({{id,opacity:conn.has(id)?1:0.07}})));
    edges.update(edges.getIds().map(id=>{{const e=edges.get(id);return{{id,color:{{opacity:(conn.has(e.from)&&conn.has(e.to))?0.9:0.04}}}};}}));
  }}else{{resetHL();}}
}});
function searchNode(q){{
  if(!q){{resetHL();return;}}q=q.toLowerCase();
  nodes.update(nodes.getIds().map(id=>({{id,opacity:id.toLowerCase().includes(q)?1:0.07}})));
  const hit=nodes.getIds().find(id=>id.toLowerCase().includes(q));
  if(hit)net.focus(hit,{{scale:1.6,animation:true}});
}}
</script></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def build_knowledge_graph(file_path: str, output_path: str) -> str:
    chunks = load_and_chunk(file_path)
    if not chunks:
        return _empty_graph(output_path, "No text could be extracted.")

    triples, domain = extract_all_triples(chunks)
    if not triples:
        return _empty_graph(output_path, "No meaningful relationships found.")

    node_meta, edges = build_graph_data(triples)
    if not edges:
        return _empty_graph(output_path, "Could not build graph.")

    return render_graph(node_meta, edges, domain, output_path)


def _empty_graph(output_path: str, reason: str = "") -> str:
    html = (f'<!DOCTYPE html><html><body style="background:#09090f;color:#888;display:flex;'
            f'align-items:center;justify-content:center;height:100vh;font-family:sans-serif;'
            f'flex-direction:column;gap:14px;"><span style="font-size:36px;">⚠️</span>'
            f'<p style="font-size:14px;">{reason}</p></body></html>')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path