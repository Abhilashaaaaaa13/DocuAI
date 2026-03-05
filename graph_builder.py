"""
graph_builder.py
Builds a clean, meaningful knowledge graph from any document.

Pipeline:
  1. smart_load  → extract text (handles PDF/scanned/image/docx/txt)
  2. Chunk text into passages → feed to LLM in batches
  3. LLM extracts (subject, relationship, object) triples directly
  4. Aggregate + deduplicate triples
  5. Render interactive vis.js HTML graph
  
Why LLM triples instead of spaCy co-occurrence:
  - spaCy picks up every noun chunk → noise (random words, page numbers, etc.)
  - LLM understands semantics → only real meaningful relationships
"""

import os
import re
import json
import spacy
import subprocess
from collections import defaultdict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from document_loader import smart_load

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Config ──
CHUNK_CHARS     = 1500   # chars per passage sent to LLM
MAX_CHUNKS      = 20     # max passages to process (keeps it fast)
TOP_NODES       = 35     # max nodes shown in graph
TOP_EDGES       = 60     # max edges shown in graph
MIN_EDGE_WEIGHT = 1      # keep even single-mention relations

# Node color by category (LLM assigns category)
CATEGORY_COLORS = {
    "technology":   "#4FC3F7",  # blue
    "organization": "#81C784",  # green
    "person":       "#FFB74D",  # orange
    "place":        "#FF8A65",  # deep orange
    "concept":      "#80DEEA",  # cyan
    "process":      "#CE93D8",  # purple
    "product":      "#F48FB1",  # pink
    "benefit":      "#A5D6A7",  # light green
    "challenge":    "#EF9A9A",  # red
    "other":        "#B0BEC5",  # grey
}


# ─────────────────────────────────────────────
#  STEP 1 — load + chunk text
# ─────────────────────────────────────────────

def load_and_chunk(file_path: str) -> list[str]:
    """Load file via smart_load, return list of text chunks."""
    print(f"[Graph] Loading: {file_path}")
    try:
        docs = smart_load(file_path)
    except Exception as e:
        print(f"[Graph] Load error: {e}")
        return []

    if not docs:
        return []

    full_text = "\n\n".join(d.page_content for d in docs)
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    # Split into chunks of ~CHUNK_CHARS
    chunks = []
    for i in range(0, len(full_text), CHUNK_CHARS):
        chunk = full_text[i : i + CHUNK_CHARS].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

    chunks = chunks[:MAX_CHUNKS]
    print(f"[Graph] {len(chunks)} text chunks to process")
    return chunks


# ─────────────────────────────────────────────
#  STEP 2 — LLM triple extraction
# ─────────────────────────────────────────────

def extract_triples_from_chunk(chunk: str, doc_topic: str) -> list[dict]:
    """
    Ask the LLM to extract (subject, relation, object, category) triples
    from a text chunk. Returns list of triple dicts.
    """
    prompt = f"""You are a knowledge graph extractor analyzing a document about: {doc_topic}

From the text below, extract the most important factual relationships.
Focus on: main topics, technologies, organizations, processes, benefits, challenges, places.
Ignore: generic words, page numbers, citations, author names.

Return ONLY a JSON array of objects. Each object must have:
  "s": subject (2-5 word concept, title case)
  "r": relationship verb (2-4 words, lowercase)
  "o": object (2-5 word concept, title case)
  "cat_s": category of subject (one of: technology/organization/person/place/concept/process/product/benefit/challenge/other)
  "cat_o": category of object

Extract 5-12 triples. Only include clear, specific, factual relationships.

Text:
{chunk}

JSON array:"""

    try:
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        # Find the JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return []
        triples = json.loads(match.group())
        # Validate shape
        valid = []
        for t in triples:
            if all(k in t for k in ["s", "r", "o"]):
                valid.append({
                    "s":     t["s"].strip().title()[:50],
                    "r":     t["r"].strip().lower()[:40],
                    "o":     t["o"].strip().title()[:50],
                    "cat_s": t.get("cat_s", "other").lower(),
                    "cat_o": t.get("cat_o", "other").lower(),
                })
        return valid
    except Exception as e:
        print(f"[Triple Extract Error] {e}")
        return []


def infer_doc_topic(chunks: list[str]) -> str:
    """Ask LLM for a 1-sentence topic summary of the document."""
    sample = chunks[0][:800] if chunks else ""
    try:
        prompt = f"""In 5-8 words, what is this document about?

Text: {sample}

Answer (just the topic, no punctuation):"""
        return llm.invoke(prompt).content.strip()
    except:
        return "the document topic"


def extract_all_triples(chunks: list[str]) -> list[dict]:
    """Extract triples from all chunks and deduplicate."""
    doc_topic = infer_doc_topic(chunks)
    print(f"[Graph] Document topic: {doc_topic}")

    all_triples = []
    for i, chunk in enumerate(chunks):
        print(f"[Graph] Extracting triples from chunk {i+1}/{len(chunks)}…")
        triples = extract_triples_from_chunk(chunk, doc_topic)
        all_triples.extend(triples)

    print(f"[Graph] Raw triples: {len(all_triples)}")
    return all_triples


# ─────────────────────────────────────────────
#  STEP 3 — aggregate into graph structure
# ─────────────────────────────────────────────

def build_graph_data(triples: list[dict]) -> tuple[dict, list]:
    """
    Returns:
      node_meta : {name: {"category": str, "freq": int}}
      edges     : [{"from": str, "to": str, "label": str, "weight": int}]
    """
    node_meta  = defaultdict(lambda: {"category": "other", "freq": 0})
    edge_count = defaultdict(lambda: {"label": "", "weight": 0})

    for t in triples:
        s, r, o = t["s"], t["r"], t["o"]

        # Skip trivially short/long nodes
        if len(s) < 2 or len(o) < 2 or len(s) > 50 or len(o) > 50:
            continue
        # Skip if subject == object
        if s.lower() == o.lower():
            continue

        node_meta[s]["category"] = t.get("cat_s", "other")
        node_meta[s]["freq"]    += 1
        node_meta[o]["category"] = t.get("cat_o", "other")
        node_meta[o]["freq"]    += 1

        key = (s, o)
        edge_count[key]["weight"] += 1
        # Keep the most recent relation label
        edge_count[key]["label"]   = r

    # Filter to top nodes by frequency
    sorted_nodes = sorted(node_meta, key=lambda n: node_meta[n]["freq"], reverse=True)
    top_node_set = set(sorted_nodes[:TOP_NODES])

    # Build final edges list
    edges = []
    for (s, o), data in sorted(edge_count.items(), key=lambda x: x[1]["weight"], reverse=True):
        if s in top_node_set and o in top_node_set:
            edges.append({
                "from":   s,
                "to":     o,
                "label":  data["label"],
                "weight": data["weight"],
            })
        if len(edges) >= TOP_EDGES:
            break

    # Only keep nodes that appear in at least one edge
    used_nodes = set()
    for e in edges:
        used_nodes.add(e["from"])
        used_nodes.add(e["to"])

    final_nodes = {n: node_meta[n] for n in used_nodes}
    return final_nodes, edges


# ─────────────────────────────────────────────
#  STEP 4 — render vis.js HTML
# ─────────────────────────────────────────────

def render_graph(node_meta: dict, edges: list, output_path: str) -> str:

    nodes_js = []
    for name, meta in node_meta.items():
        cat   = meta.get("category", "other")
        freq  = meta.get("freq", 1)
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        size  = max(16, min(50, 12 + freq * 3))
        label = name if len(name) <= 24 else name[:22] + "…"
        nodes_js.append({
            "id":    name,
            "label": label,
            "title": f"<b>{name}</b><br>Category: {cat}<br>Mentions: {freq}",
            "color": {
                "background": color,
                "border":     "#1a1a2e",
                "highlight":  {"background": "#FFD700", "border": "#FF8C00"},
                "hover":      {"background": "#FFD700", "border": "#FF8C00"},
            },
            "size":        size,
            "font":        {"color": "#f0f0f0", "size": 14, "face": "sans-serif"},
            "borderWidth": 2,
            "shadow":      {"enabled": True, "color": "rgba(0,0,0,0.6)", "size": 10},
        })

    edges_js = []
    for e in edges:
        w = e["weight"]
        edges_js.append({
            "from":   e["from"],
            "to":     e["to"],
            "label":  e["label"],
            "title":  f"{e['from']} → {e['label']} → {e['to']}",
            "width":  max(1, min(6, w * 1.5)),
            "color":  {"color": "#3a3a6a", "highlight": "#FFD700", "hover": "#aaa", "opacity": 0.85},
            "font":   {"color": "#90CAF9", "size": 11, "align": "middle", "strokeWidth": 2, "strokeColor": "#0d0d1a"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        })

    nodes_json = json.dumps(nodes_js, ensure_ascii=False)
    edges_json = json.dumps(edges_js, ensure_ascii=False)

    # Legend
    legend_html = ""
    for cat, color in CATEGORY_COLORS.items():
        if cat == "other":
            continue
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;margin:3px 10px;">'
            f'<span style="width:11px;height:11px;border-radius:50%;background:{color};'
            f'margin-right:5px;flex-shrink:0;"></span>'
            f'<span style="color:#bbb;font-size:11px;text-transform:capitalize;">{cat}</span></span>'
        )

    node_count = len(nodes_js)
    edge_count = len(edges_js)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Knowledge Graph</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
  <link  rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"/>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      background:#0d0d1a;
      font-family:'Segoe UI',sans-serif;
      color:#f0f0f0;
      height:100vh;
      display:flex;
      flex-direction:column;
      overflow:hidden;
    }}
    #topbar {{
      padding:10px 18px;
      background:linear-gradient(135deg,#1a1a2e,#16213e);
      border-bottom:1px solid #2a2a4a;
      display:flex;
      align-items:center;
      justify-content:space-between;
      flex-shrink:0;
    }}
    #topbar h2 {{ font-size:15px; color:#80DEEA; letter-spacing:1.5px; font-weight:600; }}
    #stats     {{ font-size:11px; color:#555; }}
    #legend    {{
      padding:5px 18px;
      background:#0f0f20;
      border-bottom:1px solid #1e1e3a;
      flex-shrink:0;
      display:flex;
      flex-wrap:wrap;
      align-items:center;
    }}
    #wrap {{
      flex:1;
      position:relative;
      overflow:hidden;
    }}
    #graph {{
      width:100%;
      height:100%;
      background:radial-gradient(ellipse at 50% 40%, #0f0f2a 0%, #07070f 100%);
    }}
    /* Controls */
    #controls {{
      position:absolute;
      top:12px;
      right:12px;
      display:flex;
      flex-direction:column;
      gap:5px;
    }}
    .btn {{
      background:#1a1a2e;
      border:1px solid #333;
      color:#80DEEA;
      padding:6px 13px;
      border-radius:5px;
      cursor:pointer;
      font-size:11px;
      letter-spacing:.5px;
      transition:.15s;
      white-space:nowrap;
    }}
    .btn:hover {{ background:#252545; border-color:#80DEEA; }}
    /* Search */
    #searchbox {{
      position:absolute;
      top:12px;
      left:12px;
    }}
    #searchbox input {{
      background:#1a1a2e;
      border:1px solid #333;
      color:#f0f0f0;
      padding:6px 11px;
      border-radius:5px;
      font-size:12px;
      width:200px;
      outline:none;
    }}
    #searchbox input:focus {{ border-color:#80DEEA; }}
    /* Tooltip */
    #tip {{
      position:absolute;
      bottom:12px;
      left:12px;
      background:#1a1a2eee;
      border:1px solid #2a2a4a;
      border-radius:7px;
      padding:9px 13px;
      font-size:12px;
      color:#ccc;
      max-width:300px;
      pointer-events:none;
      display:none;
      line-height:1.5;
    }}
  </style>
</head>
<body>
  <div id="topbar">
    <h2>⬡ &nbsp;Knowledge Graph</h2>
    <span id="stats">{node_count} nodes &nbsp;·&nbsp; {edge_count} edges</span>
  </div>
  <div id="legend">{legend_html}</div>
  <div id="wrap">
    <div id="graph"></div>
    <div id="searchbox">
      <input type="text" placeholder="🔍 Search node…" oninput="searchNode(this.value)"/>
    </div>
    <div id="controls">
      <button class="btn" onclick="network.fit()">⊡ &nbsp;Fit All</button>
      <button class="btn" onclick="togglePhysics()">⚙ &nbsp;Physics</button>
      <button class="btn" onclick="resetView()">↺ &nbsp;Reset</button>
    </div>
    <div id="tip"></div>
  </div>

  <script>
    const nodesData = {nodes_json};
    const edgesData = {edges_json};

    const nodes   = new vis.DataSet(nodesData);
    const edges   = new vis.DataSet(edgesData);
    const container = document.getElementById('graph');

    const options = {{
      physics: {{
        enabled: true,
        forceAtlas2Based: {{
          gravitationalConstant: -80,
          centralGravity: 0.01,
          springLength: 180,
          springConstant: 0.05,
          damping: 0.5,
          avoidOverlap: 1.0,
        }},
        solver: 'forceAtlas2Based',
        stabilization: {{ iterations: 200, updateInterval: 30 }},
        minVelocity: 0.75,
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 80,
        hideEdgesOnDrag: true,
        keyboard: {{ enabled: true }},
        zoomView: true,
        dragView: true,
      }},
      nodes: {{
        shape: 'dot',
        shadow: true,
      }},
      edges: {{
        smooth: {{ type: 'curvedCW', roundness: 0.2 }},
      }},
    }};

    const network = new vis.Network(container, {{ nodes, edges }}, options);

    // Freeze after stabilization
    let physicsOn = true;
    network.on('stabilizationIterationsDone', () => {{
      network.setOptions({{ physics: {{ enabled: false }} }});
      physicsOn = false;
      network.fit({{ animation: {{ duration: 800 }} }});
    }});

    function togglePhysics() {{
      physicsOn = !physicsOn;
      network.setOptions({{ physics: {{ enabled: physicsOn }} }});
    }}
    function resetView() {{
      nodes.update(nodes.getIds().map(id => ({{ id, opacity: 1 }})));
      network.fit({{ animation: true }});
    }}

    // Hover tooltip
    const tip = document.getElementById('tip');
    network.on('hoverNode', p => {{
      const n = nodes.get(p.node);
      tip.innerHTML = n.title;
      tip.style.display = 'block';
    }});
    network.on('blurNode',  () => {{ tip.style.display = 'none'; }});
    network.on('hoverEdge', p => {{
      const e = edges.get(p.edge);
      tip.innerHTML = e.title || '';
      tip.style.display = 'block';
    }});
    network.on('blurEdge',  () => {{ tip.style.display = 'none'; }});

    // Click → isolate neighbourhood
    network.on('click', p => {{
      if (p.nodes.length > 0) {{
        const nid = p.nodes[0];
        const connected = new Set(network.getConnectedNodes(nid));
        connected.add(nid);
        nodes.update(nodes.getIds().map(id => ({{ id, opacity: connected.has(id) ? 1 : 0.08 }})));
        edges.update(edges.getIds().map(id => {{
          const e = edges.get(id);
          return {{ id, color: {{ opacity: (connected.has(e.from) && connected.has(e.to)) ? 0.9 : 0.05 }} }};
        }}));
      }} else {{
        nodes.update(nodes.getIds().map(id => ({{ id, opacity: 1 }})));
        edges.update(edges.getIds().map(id => ({{ id, color: {{ opacity: 0.85 }} }})));
      }}
    }});

    // Search
    function searchNode(q) {{
      if (!q) {{ resetView(); return; }}
      q = q.toLowerCase();
      const updates = nodes.getIds().map(id => ({{
        id, opacity: id.toLowerCase().includes(q) ? 1 : 0.07
      }}));
      nodes.update(updates);
      const hit = nodes.getIds().find(id => id.toLowerCase().includes(q));
      if (hit) network.focus(hit, {{ scale: 1.4, animation: true }});
    }}
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def build_knowledge_graph(file_path: str, output_path: str = "knowledge_graph.html") -> str:
    """
    Full pipeline: any file → interactive HTML knowledge graph.
    Returns path to the saved HTML.
    """
    chunks = load_and_chunk(file_path)
    if not chunks:
        return _empty_graph(output_path, "No text could be extracted from this file.")

    triples = extract_all_triples(chunks)
    if not triples:
        return _empty_graph(output_path, "No meaningful relationships found in document.")

    node_meta, edges = build_graph_data(triples)
    if not edges:
        return _empty_graph(output_path, "Could not build graph — try a more content-rich document.")

    print(f"[Graph] Final graph: {len(node_meta)} nodes, {len(edges)} edges")
    return render_graph(node_meta, edges, output_path)


def _empty_graph(output_path: str, reason: str = "") -> str:
    html = f"""<!DOCTYPE html><html>
<body style="background:#0d0d1a;color:#aaa;display:flex;align-items:center;
justify-content:center;height:100vh;font-family:sans-serif;flex-direction:column;gap:12px;">
<span style="font-size:32px;">⚠️</span>
<p>{reason or 'Could not build knowledge graph.'}</p>
</body></html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path