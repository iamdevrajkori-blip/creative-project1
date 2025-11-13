
with open("streamlit_app.py", "w") as f:
    f.write('''# streamlit_app.py (enhanced graph version)
import streamlit as st
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from io import StringIO
import tempfile, json, os
import streamlit.components.v1 as components

# -------------------------
# Backend URL: replace with ngrok URL printed by backend
# -------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")

st.set_page_config(page_title="Knowledge Graph Builder", page_icon="📂", layout="wide")

# -------------------------
# Session state
# -------------------------
if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

# -------------------------
# Helpers
# -------------------------
def auth_headers():
    if st.session_state["token"]:
        return {"Authorization": f"Bearer {st.session_state['token']}"}
    return {}

def auth_params():
    if st.session_state["token"]:
        return {"token": st.session_state["token"]}
    return {}

# ----------- Upgraded Graph Visualization Functions -----------
def show_graph_from_triples(triples, title="Knowledge Graph"):
    if not triples:
        st.info("No triples to display.")
        return

    net = Network(height="600px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#222222")
    net.barnes_hut(spring_length=180, spring_strength=0.03, damping=0.8)

    # Add nodes and edges
    for s, r, o in triples:
        net.add_node(s, label=s, color="#87CEEB", shape="dot", size=18, title=s)
        net.add_node(o, label=o, color="#FFA07A", shape="dot", size=18, title=o)
        net.add_edge(s, o, title=r, label=r, color="#888888")

    net.set_options("""
    var options = {
      "edges": {
        "color": {"inherit": true},
        "smooth": {"type": "curvedCW"},
        "font": {"size": 12, "align": "middle"}
      },
      "nodes": {
        "font": {"size": 16, "color": "#222"},
        "borderWidth": 2,
        "shadow": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 180,
          "springConstant": 0.05
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "dragNodes": true,
        "zoomView": true
      }
    }
    """)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=600, scrolling=False)
    os.remove(tmp.name)


def render_pyvis_from_json(graph_json, height=600):
    """Render interactive PyVis graph from Neo4j/semantic subgraph JSON."""
    G = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#222222")
    G.barnes_hut(spring_length=180, spring_strength=0.03, damping=0.8)

    for n in graph_json.get("nodes", []):
        G.add_node(n, label=n, color="#87CEFA", shape="dot", size=16, title=n)
    for e in graph_json.get("edges", []):
        s = e.get("s")
        t = e.get("t")
        rel = e.get("rel", "")
        G.add_edge(s, t, label=rel, title=rel, color="#808080")

    G.set_options("""
    var options = {
      "edges": {
        "smooth": {"type": "dynamic"},
        "font": {"size": 12, "align": "middle"}
      },
      "nodes": {
        "font": {"size": 16, "color": "#333"},
        "borderWidth": 2,
        "shadow": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -7000,
          "centralGravity": 0.2,
          "springLength": 150,
          "springConstant": 0.04
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "navigationButtons": true
      }
    }
    """)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    G.save_graph(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=height, scrolling=False)
    os.remove(tmp.name)

# -------------------------------------------------------------------
# The rest of your original Streamlit app code remains unchanged here
# -------------------------------------------------------------------
# -------------------------
# Sidebar menu
# -------------------------
st.sidebar.title("📂 Knowledge Graph App")
menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📝 Sign Up", "🔑 Login", "👤 Profile", "📤 Upload", "📂 Datasets",
     "🧩 NER & Relations", "🔎 Graph Query", "🔎 Semantic Search & Visualization",
     "💬 Feedback System", "📊 Admin Dashboard"]
)
st.sidebar.markdown("---")
st.sidebar.write("🚀 Flask + Streamlit + Neo4j")

# -------------------------
# Home
# -------------------------
if menu == "🏠 Home":
    st.title("📂 Knowledge Graph - Milestones 2 → 4")
    st.write("Extract entities & relations, store triples to Neo4j, semantic search & subgraphs, feedback & admin.")
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=180)
    if st.session_state["username"]:
        st.info(f"Logged in as **{st.session_state['username']}**")

# -------------------------
# Sign Up
# -------------------------
elif menu == "📝 Sign Up":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username", key="signup_user")
    new_pass = st.text_input("Password", type="password", key="signup_pass")
    if st.button("Sign Up"):
        if new_user and new_pass:
            try:
                r = requests.post(f"{BASE_URL}/signup", json={"username": new_user, "password": new_pass})
                if r.status_code in (200, 201):
                    st.success(r.json().get("msg", "Account created"))
                else:
                    try:
                        st.error(r.json().get("msg", r.text))
                    except Exception:
                        st.error(f"Sign up failed: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
        else:
            st.warning("Enter both username and password")

# -------------------------
# Login
# -------------------------
elif menu == "🔑 Login":
    st.subheader("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if username and password:
            try:
                r = requests.post(f"{BASE_URL}/login", json={"username": username, "password": password})
                if r.status_code == 200:
                    res = r.json()
                    st.session_state["token"] = res.get("token")
                    st.session_state["username"] = username
                    st.success("✅ Logged in successfully!")
                else:
                    try:
                        st.error(r.json().get("msg", "Login failed"))
                    except Exception:
                        st.error(f"Login failed: HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
        else:
            st.warning("Enter both fields")

# -------------------------
# Profile
# -------------------------
elif menu == "👤 Profile":
    if st.session_state["token"]:
        headers = auth_headers()
        st.subheader("My Profile")
        if st.button("View Profile"):
            try:
                r = requests.get(f"{BASE_URL}/profile", headers=headers, params=auth_params())
                if r.status_code == 200:
                    st.json(r.json())
                else:
                    try:
                        st.error(r.json())
                    except Exception:
                        st.error(f"Failed: HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
        st.markdown("Update profile")
        lang = st.text_input("Language (comma separated)")
        interests = st.text_input("Interests (comma separated)")
        if st.button("Save Profile"):
            try:
                r = requests.post(f"{BASE_URL}/profile", headers=headers, params=auth_params(), json={"language": lang, "interests": interests})
                if r.status_code in (200, 201):
                    st.success("Profile saved")
                else:
                    st.error(r.json())
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.warning("Please login first")

# -------------------------
# Upload
# -------------------------
elif menu == "📤 Upload":
    if st.session_state["token"]:
        headers = auth_headers()
        st.subheader("Upload Dataset (CSV / JSON / TXT)")
        uploaded_file = st.file_uploader("Choose CSV, JSON, or TXT", type=["csv", "json", "txt"])
        if uploaded_file and st.button("Upload"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                r = requests.post(f"{BASE_URL}/upload", headers=headers, files=files, params=auth_params())
                if r.status_code == 200:
                    st.success(r.json().get("msg", "Uploaded"))
                else:
                    try:
                        st.error(r.json())
                    except Exception:
                        st.error(f"Upload failed: HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.warning("Please login first")

# -------------------------
# Datasets (List + Preview + Delete)
# -------------------------
elif menu == "📂 Datasets":
    if st.session_state["token"]:
        headers = auth_headers()
        st.subheader("My Datasets")
        try:
            r = requests.get(f"{BASE_URL}/datasets", headers=headers, params=auth_params())
        except Exception as e:
            st.error(f"Request failed: {e}")
            r = None

        if r and r.status_code == 200:
            datasets = r.json()
            if datasets:
                df = pd.DataFrame(datasets)
                if "size" in df.columns:
                    df["size_kb"] = (df["size"] / 1024).round(2)
                else:
                    df["size_kb"] = "-"
                cols_to_show = ["filename", "size_kb"]
                if "uploaded_at" in df.columns:
                    cols_to_show.append("uploaded_at")
                st.dataframe(df[cols_to_show], use_container_width=True)

                for idx, ds in enumerate(datasets):
                    with st.expander(f"📄 {ds['filename']}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("👀 Preview", key=f"prev_{idx}_{ds['filename']}"):
                                try:
                                    prev_res = requests.get(f"{BASE_URL}/datasets/preview/{ds['filename']}", headers=headers, params=auth_params())
                                except Exception as e:
                                    st.error(f"Request failed: {e}")
                                    prev_res = None
                                if prev_res and prev_res.status_code == 200:
                                    payload = prev_res.json()
                                    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                                        st.dataframe(pd.DataFrame(payload))
                                    else:
                                        st.json(payload)
                                else:
                                    try:
                                        st.error(prev_res.json())
                                    except Exception:
                                        st.error("Preview failed")
                        with c2:
                            if st.button("🗑️ Delete", key=f"del_{idx}_{ds['filename']}"):
                                try:
                                    del_res = requests.delete(f"{BASE_URL}/datasets/{ds['filename']}", headers=headers, params=auth_params())
                                except Exception as e:
                                    st.error(f"Request failed: {e}")
                                    del_res = None
                                if del_res and del_res.status_code == 200:
                                    st.success(del_res.json().get("msg","Deleted"))
                                    st.experimental_rerun()
                                else:
                                    try:
                                        st.error(del_res.json())
                                    except Exception:
                                        st.error("Delete failed")
            else:
                st.info("ℹ️ No datasets uploaded")
        else:
            st.error("Failed to fetch datasets")
    else:
        st.warning("Please login first")

# -------------------------
# Entity & Relation Extraction (Text input + Dataset-driven)
# -------------------------
elif menu == "🧩 NER & Relations":
    st.subheader("Entity & Relation Extraction")

    if st.session_state["token"]:
        headers = auth_headers()
        col1, col2 = st.columns(2)

        # ----- Manual Text Extraction -----
        with col1:
            st.markdown("### 🔤 Text Input")
            text_input = st.text_area(
                "Enter text for extraction",
                "Barack Obama was born in Honolulu. He served as the 44th President of the United States."
            )
            if st.button("Extract from Text"):
                if not text_input.strip():
                    st.warning("Enter some text to extract from")
                else:
                    try:
                        r = requests.post(
                            f"{BASE_URL}/extract",
                            headers=headers,
                            params={"token": st.session_state["token"]},
                            json={"text": text_input}
                        )
                        if r.status_code == 200:
                            res = r.json()
                            st.markdown("**Entities (spaCy):**")
                            st.json(res.get("entities", []))
                            st.markdown("**Parsed Triples:**")
                            triples = res.get("triples", [])
                            st.json(triples)
                            st.markdown("**Stored in Neo4j:**")
                            st.json(res.get("stored", []))
                            if triples:
                                show_graph_from_triples(triples, title="Extracted Graph from Text")
                        else:
                            st.error("Extraction failed")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

        # ----- Dataset-driven Extraction -----
        with col2:
            st.markdown("### 📂 Dataset Input")
            try:
                r = requests.get(f"{BASE_URL}/datasets", headers=headers, params={"token": st.session_state["token"]})
                if r.status_code == 200:
                    datasets = r.json()
                    if datasets:
                        selected_ds = st.selectbox("Select a dataset", [ds["filename"] for ds in datasets])
                        if st.button("Extract from Dataset"):
                            try:
                                prev_res = requests.get(f"{BASE_URL}/datasets/preview/{selected_ds}", headers=headers, params={"token": st.session_state["token"]})
                                if prev_res.status_code != 200:
                                    st.error("Could not preview file for extraction")
                                else:
                                    payload = prev_res.json()
                                    sample_text = ""
                                    if isinstance(payload, list) and payload:
                                        if isinstance(payload[0], dict):
                                            for row in payload:
                                                sample_text += " ".join([str(v) for v in row.values()]) + " "
                                        else:
                                            sample_text = " ".join([str(x) for x in payload])
                                    elif isinstance(payload, dict):
                                        sample_text = " ".join([f"{k} {v}" for k, v in payload.items()])
                                    else:
                                        sample_text = str(payload)

                                    if not sample_text.strip():
                                        st.warning("No usable text in dataset preview to extract from")
                                    else:
                                        ext_res = requests.post(f"{BASE_URL}/extract", headers=headers, params={"token": st.session_state["token"]}, json={"text": sample_text})
                                        if ext_res.status_code == 200:
                                            res = ext_res.json()
                                            st.markdown("**Entities (spaCy):**")
                                            st.json(res.get("entities", []))
                                            st.markdown("**Parsed Triples:**")
                                            triples = res.get("triples", [])
                                            st.json(triples)
                                            st.markdown("**Stored in Neo4j:**")
                                            st.json(res.get("stored", []))
                                            if triples:
                                                show_graph_from_triples(triples, title=f"Graph for {selected_ds}")
                                        else:
                                            st.error("Extraction failed")
                            except Exception as e:
                                st.error(f"Request failed: {e}")
                    else:
                        st.info("No datasets uploaded")
                else:
                    st.error("Failed to fetch datasets")
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.warning("Please login first")

# -------------------------
# Graph Query
# -------------------------
elif menu == "🔎 Graph Query":
    st.subheader("Query Knowledge Graph")
    if not st.session_state["token"]:
        st.warning("Please login first")
    else:
        headers = auth_headers()
        q_entity = st.text_input("Entity to query (exact name)", "")
        if st.button("Query Entity"):
            try:
                r = requests.get(f"{BASE_URL}/graph/{q_entity}", headers=headers, params=auth_params())
            except Exception as e:
                st.error(f"Request failed: {e}")
                r = None
            if r and r.status_code == 200:
                rows = r.json()
                st.json(rows)
                if rows:
                    triples = [(rec["a"], rec["rel"], rec["b"]) for rec in rows]
                    show_graph_from_triples(triples, title=f"Graph for {q_entity}")
            else:
                if r is not None:
                    try:
                        st.error(r.json())
                    except Exception:
                        st.error(f"Query failed: HTTP {r.status_code}")

        if st.button("Load All Relations"):
            try:
                r = requests.get(f"{BASE_URL}/graph/all", headers=headers, params=auth_params())
            except Exception as e:
                st.error(f"Request failed: {e}")
                r = None
            if r and r.status_code == 200:
                rows = r.json()
                st.json(rows)
                if rows:
                    triples = [(rec["a"], rec["rel"], rec["b"]) for rec in rows]
                    show_graph_from_triples(triples, title="All Relations")
            else:
                if r is not None:
                    try:
                        st.error(r.json())
                    except Exception:
                        st.error("Failed to load graph")

# -------------------------
# Milestone 3: Semantic Search & Visualization
# -------------------------
elif menu == "🔎 Semantic Search & Visualization":
    st.subheader("Semantic Search → Search-driven Subgraph")

    if not st.session_state["token"]:
        st.warning("Please login first")
    else:
        headers = auth_headers()
        c1, c2 = st.columns([3,1])
        with c1:
            query = st.text_input("Enter query (semantic):", "")
        with c2:
            top_k = st.number_input("Top-k", min_value=1, max_value=10, value=3)
            radius = st.slider("k-hop radius", 1, 3, 1)
        if st.button("Run Semantic Search"):
            if not query.strip():
                st.warning("Enter a query")
            else:
                payload = {"query": query, "top_k": int(top_k)}
                try:
                    r = requests.post(f"{BASE_URL}/semantic-search", headers=headers, params=auth_params(), json=payload)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    r = None
                if r and r.status_code == 200:
                    res = r.json()
                    results = res.get("results", [])
                    if results:
                        df = pd.DataFrame(results)
                        st.subheader("Top matches")
                        st.dataframe(df)
                    else:
                        st.info("No matches found")
                else:
                    if r is not None:
                        try:
                            st.error(r.json())
                        except Exception:
                            st.error("Semantic search failed")

        st.markdown("---")
        st.markdown("### Generate search-driven subgraph (union of ego graphs)")
        q2 = st.text_input("Query for subgraph (you can reuse the above):", "")
        col_a, col_b = st.columns(2)
        with col_a:
            use_prev = st.checkbox("Use previous query results", value=True)
        with col_b:
            download_name = st.text_input("Download filename", "subgraph.json")

        if st.button("Generate Subgraph"):
            use_query = (query if use_prev and query.strip() else q2).strip()
            if not use_query:
                st.warning("Enter a query for subgraph")
            else:
                payload = {"query": use_query, "top_k": int(top_k), "radius": int(radius)}
                try:
                    r = requests.post(f"{BASE_URL}/semantic-subgraph", headers=headers, params=auth_params(), json=payload)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    r = None
                if r and r.status_code == 200:
                    data = r.json()
                    st.subheader("Subgraph Summary")
                    st.write(f"Top nodes: {data.get('top_nodes', [])}")
                    st.write(f"Scores: {data.get('scores', [])}")
                    st.write(f"Nodes: {len(data.get('nodes', []))}, Edges: {len(data.get('edges', []))}")
                    # render pyvis
                    html = render_pyvis_from_json(data)
                    st.components.v1.html(html, height=600, scrolling=True)
                    # download JSON
                    json_str = json.dumps(data, indent=2)
                    st.download_button("⬇️ Download Subgraph JSON", json_str, download_name, "application/json")
                else:
                    if r is not None:
                        try:
                            st.error(r.json())
                        except Exception:
                            st.error("Subgraph generation failed")

# -------------------------
# Milestone 4 – Feedback System (User)
# -------------------------
elif menu == "💬 Feedback System":
    st.subheader("💬 Submit Feedback")
    if not st.session_state["token"]:
        st.warning("Please login first")
    else:
        query = st.text_input("Query or feature you used:")
        rating = st.slider("Rate the accuracy (1 – 5)", 1, 5, 3)
        comments = st.text_area("Comments:")
        if st.button("Submit Feedback"):
            payload = {"query": query, "rating": rating, "comments": comments}
            try:
                r = requests.post(f"{BASE_URL}/feedback",
                                  headers=auth_headers(),
                                  params=auth_params(),
                                  json=payload)
                if r.status_code == 200:
                    st.success("✅ Feedback recorded successfully!")
                else:
                    try:
                        st.error(r.json())
                    except Exception:
                        st.error(f"Failed: HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# -------------------------
# Milestone 4 – Admin Dashboard
# -------------------------
elif menu == "📊 Admin Dashboard":
    st.subheader("📊 Admin Dashboard (user ID 1 only)")
    if not st.session_state["token"]:
        st.warning("Please login first")
    else:
        try:
            res = requests.get(f"{BASE_URL}/admin/stats", headers=auth_headers(), params=auth_params())
            if res.status_code != 200:
                st.error(res.json().get("error", "Unauthorized or failed"))
            else:
                stats = res.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Entities", stats["nodes"])
                c2.metric("Total Relations", stats["relations"])
                c3.metric("Feedback Entries", stats["feedbacks"])

                # ---- All Feedback ----
                st.markdown("---")
                st.subheader("📋 All Feedback Entries")
                fb = requests.get(f"{BASE_URL}/admin/feedbacks", headers=auth_headers(), params=auth_params())
                if fb.status_code == 200:
                    data = fb.json()
                    if data:
                        st.dataframe(pd.DataFrame(data))
                    else:
                        st.info("No feedback yet.")
                else:
                    st.error(fb.text)

                # ---- View All Triples ----
                st.markdown("---")
                st.subheader("📘 View All Triples (Neo4j)")
                if st.button("Load All Triples"):
                    try:
                        triples_res = requests.get(f"{BASE_URL}/admin/triples", headers=auth_headers(), params=auth_params())
                        if triples_res.status_code == 200:
                            triples_data = triples_res.json()
                            if triples_data:
                                st.success(f"Loaded {len(triples_data)} triples.")
                                df_triples = pd.DataFrame(triples_data)
                                st.dataframe(df_triples, use_container_width=True)
                            else:
                                st.info("No triples found in Neo4j.")
                        else:
                            st.error(triples_res.text)
                    except Exception as e:
                        st.error(f"Request failed: {e}")                  

                # ---- Add Triple ----
                st.markdown("---")
                st.subheader("🧱 Add Triple to Knowledge Base")
                e1 = st.text_input("Entity 1")
                rel = st.text_input("Relation")
                e2 = st.text_input("Entity 2")
                if st.button("Add Triple"):
                    payload = {"entity1": e1, "relation": rel, "entity2": e2}
                    add = requests.post(f"{BASE_URL}/kb/add", headers=auth_headers(),
                                        params=auth_params(), json=payload)
                    if add.status_code == 200:
                        st.success("Triple added successfully!")
                    else:
                        st.error(add.text)

                # ---- Delete Triple ----
                st.subheader("🗑 Delete Triple from Knowledge Base")
                e1d = st.text_input("Entity 1 (del)")
                reld = st.text_input("Relation (del)")
                e2d = st.text_input("Entity 2 (del)")
                if st.button("Delete Triple"):
                    payload = {"entity1": e1d, "relation": reld, "entity2": e2d}
                    rem = requests.post(f"{BASE_URL}/kb/delete", headers=auth_headers(),
                                        params=auth_params(), json=payload)
                    if rem.status_code == 200:
                        st.success("Triple deleted successfully!")
                    else:
                        st.error(rem.text)

                # ---- Edit Triple ----
                st.markdown("---")
                st.subheader("✏️ Edit Existing Triple")
                old_e1 = st.text_input("Old Entity 1")
                old_rel = st.text_input("Old Relation")
                old_e2 = st.text_input("Old Entity 2")
                new_e1 = st.text_input("New Entity 1")
                new_rel = st.text_input("New Relation")
                new_e2 = st.text_input("New Entity 2")
                if st.button("Update Triple"):
                    payload = {
                        "old_entity1": old_e1, "old_relation": old_rel, "old_entity2": old_e2,
                        "new_entity1": new_e1, "new_relation": new_rel, "new_entity2": new_e2
                    }
                    edit = requests.post(f"{BASE_URL}/kb/edit", headers=auth_headers(),
                                         params=auth_params(), json=payload)
                    if edit.status_code == 200:
                        st.success("Triple updated successfully!")
                    else:
                        st.error(edit.text)

                # ---- Feedback Analytics ----
                st.markdown("---")
                st.subheader("📈 Feedback Analytics")
                if 'data' in locals() and isinstance(data, list) and data:
                    df_fb = pd.DataFrame(data)
                    if not df_fb.empty and "rating" in df_fb.columns:
                        avg_rating = df_fb["rating"].mean().round(2)
                        st.metric("Average Rating", avg_rating)
                        st.bar_chart(df_fb["rating"].value_counts().sort_index())
                    else:
                        st.info("No ratings yet.")
                else:
                    st.info("No feedback data loaded.")

        except Exception as e:
            st.error(f"Request failed: {e}")











 ''')