# backend.py
from flask import Flask, request, jsonify
import sqlite3, bcrypt, jwt, datetime, os, json, re
from pyngrok import ngrok
from threading import Thread
import pandas as pd
import spacy
from transformers import pipeline
from neo4j import GraphDatabase
from functools import wraps
import time
import numpy as np

# Sentence transformers imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print("Sentence-transformers / sklearn import error:", e)
    SentenceTransformer = None
    cosine_similarity = None

# -------------------------
# Config / Setup
# -------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"
SECRET = app.config.get("SECRET_KEY", "mysecret")
UPLOAD_FOLDER = "data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SQLite DB init (users + uploads + profiles)
con = sqlite3.connect("users.db")
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password BLOB)")
cur.execute("CREATE TABLE IF NOT EXISTS profiles(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, language TEXT, interests TEXT)")
cur.execute("CREATE TABLE IF NOT EXISTS uploads(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, filename TEXT, filepath TEXT)")
con.commit()
con.close()

# Feedback DB init (Milestone 4)
con = sqlite3.connect("feedback.db")
cur = con.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    query TEXT,
    rating INTEGER,
    comments TEXT,
    timestamp TEXT
)
""")
con.commit()
con.close()

# Neo4j config (kept as requested)
NEO4J_URI = "neo4j+s://211ec986.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Ma76ZlxSZQZaXcd6zsbdLiNowrRy3o2XOXCgweQkoWk"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("spaCy load error:", e)
    nlp = None

# Load Rebel model (relation extraction) - optional (may be heavy)
try:
    re_model = pipeline("text2text-generation", model="Babelscape/rebel-large")
except Exception as e:
    print("Rebel model load error:", e)
    re_model = None

# -------------------------
# Utils
# -------------------------
def sanitize_rel(rel: str) -> str:
    rel = re.sub(r"[^A-Za-z0-9 ]+", " ", str(rel))
    rel = re.sub(r"\s+", "_", rel.strip()).upper()
    if not rel:
        rel = "RELATED_TO"
    if re.match(r"^\d", rel):
        rel = "R_" + rel
    return rel

def store_triple(e1, rel, e2):
    rel_name = sanitize_rel(rel)
    query = (
        f"MERGE (a:Entity {{name: $e1}}) "
        f"MERGE (b:Entity {{name: $e2}}) "
        f"MERGE (a)-[r:{rel_name}]->(b) "
        "RETURN a.name AS a_name, type(r) AS rel_type, b.name AS b_name"
    )
    with driver.session() as session:
        result = session.run(query, e1=e1, e2=e2)
        return [{"a": rec["a_name"], "rel": rec["rel_type"], "b": rec["b_name"]} for rec in result]

def parse_rebel_output(raw_output):
    triples = []
    # Rebel returns list of dicts with generated_text or similar
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        text = raw_output[0].get("generated_text", "").strip()
    else:
        text = str(raw_output)

    # Rebel formatting may vary; attempt a robust parse:
    # split on double spaces or newline groups
    parts = [p.strip() for p in re.split(r"\n+|\s\s+", text) if p.strip()]
    # If format looks like subj, obj, rel repeating every 3 parts:
    if len(parts) >= 3:
        for i in range(0, len(parts) - 2, 3):
            subj = parts[i]
            obj = parts[i+1]
            rel = parts[i+2]
            triples.append((subj, rel, obj))
    else:
        # fallback naive extraction: try splitting by '|' or ';'
        alt = [p.strip() for p in re.split(r"\||;", text) if p.strip()]
        if len(alt) >= 3:
            for i in range(0, len(alt) - 2, 3):
                triples.append((alt[i], alt[i+2], alt[i+1]))
    return triples

def preview_file(filepath):
    try:
        if filepath.lower().endswith(".csv"):
            try:
                df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(filepath, encoding="latin1", on_bad_lines="skip")
            return df.head().to_dict(orient="records")
        elif filepath.lower().endswith(".json"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                return data[:5] if isinstance(data, list) else dict(list(data.items())[:5])
        elif filepath.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for _ in range(5):
                    try:
                        lines.append(next(f).rstrip("\n"))
                    except StopIteration:
                        break
                return lines
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Preview not available"}

# -------------------------
# Token helper
# -------------------------
def _extract_token_from_request():
    """Return token string if present (header, args, form, or json)."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    # Accept "Bearer <token>" or just token
    if auth:
        parts = auth.split()
        if len(parts) >= 2:
            return parts[1]
        return parts[-1]

    token = request.args.get("token") or request.form.get("token")
    if token:
        return token

    # finally check JSON body key "token"
    try:
        data = request.get_json(silent=True) or {}
        if isinstance(data, dict):
            return data.get("token")
    except Exception:
        pass

    return None

def check_token(f):
    """Decorator: extracts token, decodes it and on success calls f(user_id, *args, **kwargs)."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = _extract_token_from_request()
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])
            user_id = payload.get("user_id")
            if not user_id:
                return jsonify({"error": "Token invalid"}), 401
            return f(user_id, *args, **kwargs)
        except jwt.ExpiredSignatureError as e:
            return jsonify({"error": "Token expired"}), 401
        except Exception as e:
            # try fallback param token if different (helpful in ngrok flows)
            fallback = request.args.get("token")
            if fallback and fallback != token:
                try:
                    payload = jwt.decode(fallback, SECRET, algorithms=["HS256"])
                    return f(payload.get("user_id"), *args, **kwargs)
                except Exception:
                    pass
            return jsonify({"error": "Token invalid", "details": str(e)}), 401
    return wrapper

# -------------------------
# Auth Routes
# -------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return {"msg": "username and password required"}, 400
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        con = sqlite3.connect("users.db")
        cur = con.cursor()
        cur.execute("INSERT INTO users(username,password) VALUES(?,?)", (username, hashed))
        con.commit()
        con.close()
        return {"msg": "user created"}
    except Exception:
        return {"msg": "username already exists"}, 400

@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return {"msg": "username and password required"}, 400

    con = sqlite3.connect("users.db")
    cur = con.cursor()
    cur.execute("SELECT id,password FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    con.close()
    if row and bcrypt.checkpw(password.encode(), row[1]):
        token = jwt.encode(
            {"user_id": row[0], "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)},
            SECRET, algorithm="HS256"
        )
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return {"msg": "login ok", "token": token}
    return {"msg": "wrong username or password"}, 401

# -------------------------
# Profile
# -------------------------
@app.route("/profile", methods=["GET","POST"])
@check_token
def profile(user_id):
    con = sqlite3.connect("users.db")
    cur = con.cursor()
    if request.method == "POST":
        data = request.json or {}
        lang = data.get("language", "English")
        intr = data.get("interests", "")
        cur.execute("DELETE FROM profiles WHERE user_id=?", (user_id,))
        cur.execute("INSERT INTO profiles(user_id,language,interests) VALUES(?,?,?)", (user_id, lang, intr))
        con.commit()
        con.close()
        return {"msg": "profile saved"}
    else:
        cur.execute("SELECT language,interests FROM profiles WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        con.close()
        return {"language": row[0], "interests": row[1]} if row else {}

# -------------------------
# Upload & Datasets
# -------------------------
@app.route("/upload", methods=["POST"])
@check_token
def upload(user_id):
    if "file" not in request.files:
        return {"msg":"no file"}, 400
    file = request.files["file"]
    user_folder = os.path.join(UPLOAD_FOLDER, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    path = os.path.join(user_folder, file.filename)
    with open(path, "wb") as f:
        f.write(file.read())
    con = sqlite3.connect("users.db")
    cur = con.cursor()
    cur.execute("INSERT INTO uploads(user_id,filename,filepath) VALUES(?,?,?)", (user_id, file.filename, path))
    con.commit()
    con.close()
    return {"msg": f"{file.filename} uploaded"}

@app.route("/datasets", methods=["GET"])
@check_token
def list_datasets(user_id):
    con = sqlite3.connect("users.db")
    cur = con.cursor()
    cur.execute("SELECT filename, filepath FROM uploads WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    con.close()
    datasets = []
    for f, path in rows:
        try:
            size = os.path.getsize(path)
            uploaded_at = datetime.datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            size = 0
            uploaded_at = "-"
        datasets.append({"filename": f, "size": size, "uploaded_at": uploaded_at})
    return jsonify(datasets)

@app.route("/datasets/preview/<filename>", methods=["GET"])
@check_token
def preview_dataset(user_id, filename):
    con = sqlite3.connect("users.db")
    cur = con.cursor()
    cur.execute("SELECT filepath FROM uploads WHERE user_id=? AND filename=?", (user_id, filename))
    row = cur.fetchone()
    con.close()
    if not row:
        return {"error": "File not found"}, 404
    filepath = row[0]
    return jsonify(preview_file(filepath))

@app.route("/datasets/<filename>", methods=["DELETE"])
@check_token
def delete_dataset(user_id, filename):
    con = sqlite3.connect("users.db")
    cur = con.cursor()
    cur.execute("SELECT filepath FROM uploads WHERE user_id=? AND filename=?", (user_id, filename))
    row = cur.fetchone()
    if row:
        filepath = row[0]
        if os.path.exists(filepath):
            os.remove(filepath)
        cur.execute("DELETE FROM uploads WHERE user_id=? AND filename=?", (user_id, filename))
        con.commit()
        con.close()
        return {"msg": f"{filename} deleted"}
    con.close()
    return {"error": "File not found"}, 404

# -------------------------
# Extract (NER + RE)
# -------------------------
@app.route("/extract", methods=["POST"])
@check_token
def extract(user_id):
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}, 400

    doc = nlp(text) if nlp else None
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents] if doc else []

    raw_relations = re_model(text, max_length=512, truncation=True) if re_model else []
    triples = parse_rebel_output(raw_relations)

    stored = []
    for s,r,o in triples:
        try:
            stored.extend(store_triple(s, r, o))
        except Exception as e:
            stored.append({"error": str(e), "triple": (s,r,o)})

    return {"entities": entities, "triples": triples, "stored": stored, "raw_relations": raw_relations}

# -------------------------
# Graph Query
# -------------------------
@app.route("/graph/<entity>", methods=["GET"])
@check_token
def query_graph(user_id, entity):
    query = "MATCH (a:Entity {name:$entity})-[r]->(b) RETURN a.name AS a_name, type(r) AS rel_type, b.name AS b_name"
    with driver.session() as session:
        result = session.run(query, entity=entity)
        return jsonify([{"a": rec["a_name"], "rel": rec["rel_type"], "b": rec["b_name"]} for rec in result])

@app.route("/graph/all", methods=["GET"])
@check_token
def graph_all(user_id):
    query = "MATCH (a)-[r]->(b) RETURN a.name AS a_name, type(r) AS rel_type, b.name AS b_name LIMIT 500"
    with driver.session() as session:
        result = session.run(query)
        return jsonify([{"a": rec["a_name"], "rel": rec["rel_type"], "b": rec["b_name"]} for rec in result])

# -------------------------
# Semantic embeddings + subgraphs (in-memory cache)
# -------------------------
NODE_LIST = []
NODE_EMBS = None
EMB_MODEL = None
EMB_LAST_REFRESH = 0
EMB_REFRESH_SECONDS = 300  # refresh every 5 minutes

def refresh_node_embeddings(force=False):
    global NODE_LIST, NODE_EMBS, EMB_MODEL, EMB_LAST_REFRESH
    now = time.time()
    if not force and (now - EMB_LAST_REFRESH) < EMB_REFRESH_SECONDS and NODE_LIST:
        return  # still fresh
    # fetch nodes from neo4j
    try:
        with driver.session() as session:
            res = session.run("MATCH (n:Entity) RETURN DISTINCT n.name AS name")
            nodes = [r["name"] for r in res]
    except Exception as e:
        print("Neo4j fetch nodes error:", e)
        nodes = []
    NODE_LIST = nodes
    if SentenceTransformer is None:
        print("SentenceTransformer not available - cannot compute embeddings")
        NODE_EMBS = None
        EMB_LAST_REFRESH = now
        return
    try:
        if EMB_MODEL is None:
            EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        if NODE_LIST:
            NODE_EMBS = EMB_MODEL.encode(NODE_LIST, convert_to_numpy=True, show_progress_bar=False)
        else:
            NODE_EMBS = None
    except Exception as e:
        print("Embedding error:", e)
        NODE_EMBS = None
    EMB_LAST_REFRESH = now
    print(f"Embeddings refreshed: {len(NODE_LIST)} nodes, time={EMB_LAST_REFRESH}")

@app.route("/graph/nodes", methods=["GET"])
@check_token
def graph_nodes(user_id):
    try:
        with driver.session() as session:
            res = session.run("MATCH (n:Entity) RETURN DISTINCT n.name AS name")
            nodes = [r["name"] for r in res]
    except Exception as e:
        print("graph_nodes error:", e)
        nodes = []
    return jsonify({"nodes": nodes})

@app.route("/graph/subgraph/<entity>", methods=["GET"])
@check_token
def graph_subgraph(user_id, entity):
    radius = int(request.args.get("radius", 1))
    cypher = """
    MATCH p=(start:Entity {name:$name})-[*1..$k]-(m)
    WITH collect(distinct nodes(p)) AS nds, collect(distinct relationships(p)) AS rels
    UNWIND nds AS nd_list
    UNWIND nd_list AS nd
    WITH collect(distinct nd.name) AS node_names, rels
    UNWIND rels AS rlist
    UNWIND rlist AS rr
    RETURN node_names AS nodes,
           collect(distinct {s: startNode(rr).name, t: endNode(rr).name, rel: type(rr)}) AS edges
    """
    try:
        with driver.session() as session:
            res = session.run(cypher, name=entity, k=radius)
            rec = res.single()
            if not rec:
                return jsonify({"nodes": [], "edges": []})
            nodes = rec["nodes"] or []
            edges = rec["edges"] or []
        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        print("graph_subgraph error:", e)
        return jsonify({"nodes": [], "edges": [], "error": str(e)}), 500

@app.route("/semantic-search", methods=["POST"])
@check_token
def semantic_search(user_id):
    payload = request.json or {}
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 3))
    if not query:
        return jsonify({"error": "No query provided"}), 400
    refresh_node_embeddings()
    if NODE_EMBS is None or EMB_MODEL is None:
        return jsonify({"error": "Embeddings not available on server"}), 500
    try:
        q_emb = EMB_MODEL.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, NODE_EMBS)[0]  # shape (N,)
        idx = np.argsort(sims)[::-1][:top_k]
        results = [{"node": NODE_LIST[i], "score": float(sims[i])} for i in idx]
        return jsonify({"results": results})
    except Exception as e:
        print("Semantic search error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/semantic-subgraph", methods=["POST"])
@check_token
def semantic_subgraph(user_id):
    payload = request.json or {}
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 3))
    radius = int(payload.get("radius", 1))
    if not query:
        return jsonify({"error": "No query provided"}), 400

    refresh_node_embeddings()
    if NODE_EMBS is None or EMB_MODEL is None:
        return jsonify({"error": "Embeddings not available on server"}), 500

    # semantic search
    try:
        q_emb = EMB_MODEL.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, NODE_EMBS)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        top_nodes = [NODE_LIST[i] for i in idx]
        top_scores = [float(sims[i]) for i in idx]
    except Exception as e:
        print("Semantic search error:", e)
        return jsonify({"error": str(e)}), 500

    # subgraph expansion
    all_nodes = set()
    all_edges = set()
    cypher_template = """
    MATCH p=(start:Entity {{name:$name}})-[*1..{k}]-(m)
    WITH collect(distinct nodes(p)) AS nds, collect(distinct relationships(p)) AS rels
    UNWIND nds AS nd_list
    UNWIND nd_list AS nd
    WITH collect(distinct nd.name) AS node_names, rels
    UNWIND rels AS rlist
    UNWIND rlist AS rr
    RETURN node_names AS nodes,
           collect(distinct {{s: startNode(rr).name, t: endNode(rr).name, rel: type(rr)}}) AS edges
    """.format(k=radius)

    try:
        with driver.session() as session:
            for n in top_nodes:
                res = session.run(cypher_template, {"name": n})
                rec = res.single()
                if not rec:
                    continue
                nodes = rec["nodes"] or []
                edges = rec["edges"] or []
                for nd in nodes:
                    all_nodes.add(nd)
                for e in edges:
                    s = e.get("s")
                    t = e.get("t")
                    rel = e.get("rel")
                    if s and t:
                        all_edges.add((s, t, rel))
        if not all_nodes:
            all_nodes.update(top_nodes)
        nodes_list = sorted(list(all_nodes))
        edges_list = [{"s": s, "t": t, "rel": rel} for (s, t, rel) in sorted(all_edges)]
        return jsonify({
            "nodes": nodes_list,
            "edges": edges_list,
            "top_nodes": top_nodes,
            "scores": top_scores
        })
    except Exception as e:
        print("Neo4j subgraph error:", type(e), e)
        return jsonify({"error": "Neo4j query failed", "details": str(e)}), 500

# -------------------------
# Milestone 4: Feedback System & Admin routes
# -------------------------
@app.route("/feedback", methods=["POST"])
@check_token
def feedback_route(user_id):
    data = request.json or {}
    query = data.get("query", "")
    rating = int(data.get("rating", 3))
    comments = data.get("comments", "")
    con = sqlite3.connect("feedback.db")
    cur = con.cursor()
    cur.execute(
        "INSERT INTO feedback(user_id,query,rating,comments,timestamp) VALUES(?,?,?,?,datetime('now'))",
        (user_id, query, rating, comments)
    )
    con.commit()
    con.close()
    return {"msg": "Feedback recorded"}

@app.route("/admin/feedbacks", methods=["GET"])
@check_token
def admin_feedbacks(user_id):
    if user_id != 1:
        return {"error": "Admin only"}, 403
    con = sqlite3.connect("feedback.db")
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", con)
    con.close()
    return df.to_dict(orient="records")

@app.route("/admin/triples", methods=["GET"])
@check_token
def admin_triples(user_id):
    """Return all triples from Neo4j for admin view."""
    if user_id != 1:
        return {"error": "Admin only"}, 403
    try:
        with driver.session() as session:
            res = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.name AS entity1, type(r) AS relation, b.name AS entity2
                LIMIT 1000
            """)
            triples = [
                {"entity1": rec["entity1"], "relation": rec["relation"], "entity2": rec["entity2"]}
                for rec in res
            ]
        return jsonify(triples)
    except Exception as e:
        print("admin_triples error:", e)
        return jsonify({"error": str(e)}), 500  

@app.route("/admin/stats", methods=["GET"])
@check_token
def admin_stats(user_id):
    if user_id != 1:
        return {"error": "Admin only"}, 403
    try:
        with driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    except Exception as e:
        print("admin_stats neo4j error:", e)
        nodes, rels = 0, 0
    con = sqlite3.connect("feedback.db")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    feedbacks = cur.fetchone()[0]
    con.close()
    return jsonify({"nodes": nodes, "relations": rels, "feedbacks": feedbacks})

# Admin KB CRUD
@app.route("/kb/add", methods=["POST"])
@check_token
def kb_add(user_id):
    if user_id != 1:
        return {"error": "Admin only"}, 403
    data = request.json or {}
    e1, rel, e2 = data.get("entity1"), data.get("relation"), data.get("entity2")
    if not e1 or not e2 or not rel:
        return {"error": "Missing fields"}, 400
    stored = store_triple(e1, rel, e2)
    return jsonify({"msg": "Triple added", "stored": stored})

@app.route("/kb/edit", methods=["POST"])
@check_token
def kb_edit(user_id):
    """Admin edit (update) an existing triple."""
    if user_id != 1:
        return {"error": "Admin only"}, 403

    data = request.json or {}
    old_e1, old_rel, old_e2 = data.get("old_entity1"), data.get("old_relation"), data.get("old_entity2")
    new_e1, new_rel, new_e2 = data.get("new_entity1"), data.get("new_relation"), data.get("new_entity2")

    if not all([old_e1, old_rel, old_e2, new_e1, new_rel, new_e2]):
        return {"error": "Missing fields"}, 400

    rel_old = sanitize_rel(old_rel)
    rel_new = sanitize_rel(new_rel)
    try:
        with driver.session() as session:
            # Delete the old relation
            session.run(f"""
                MATCH (a:Entity {{name:$old_e1}})-[r:{rel_old}]->(b:Entity {{name:$old_e2}})
                DELETE r
            """, old_e1=old_e1, old_e2=old_e2)

            # Create the new relation
            session.run(f"""
                MERGE (a:Entity {{name:$new_e1}})
                MERGE (b:Entity {{name:$new_e2}})
                MERGE (a)-[r:{rel_new}]->(b)
            """, new_e1=new_e1, new_e2=new_e2)

        return {"msg": "Triple updated successfully"}
    except Exception as e:
        return {"error": str(e)}, 500



# -------------------------
# Run (Docker / Local)
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


