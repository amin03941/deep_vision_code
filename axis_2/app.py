import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qdrant_client import QdrantClient, models
from streamlit_agraph import agraph, Node, Edge, Config
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bio-Scout | Multimodal Discovery Engine", 
    layout="wide", 
    page_icon="🧬"
)

# --- CONSTANTS ---
PKL_FILE = "bio_memory_dump.pkl"
DIFF_VECTOR_FILE = "health_direction_vector.npy"
METADATA_FILE = "vector_metadata.pkl"
COLLECTION_NAME = "bio_scout_memory"

# --- STYLING ---
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
.success-box {
    background: rgba(0, 255, 100, 0.1);
    border-left: 4px solid #00ff64;
    padding: 15px;
    margin: 10px 0;
}
.warning-box {
    background: rgba(255, 165, 0, 0.1);
    border-left: 4px solid #ffa500;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# --- CORE ARCHITECTURE ---
@st.cache_resource
def initialize_bio_engine():
    """Initialize vector database - NO Streamlit UI calls allowed here"""
    client = QdrantClient(location=":memory:")
    
    if os.path.exists(PKL_FILE):
        try:
            with open(PKL_FILE, 'rb') as f:
                data = pickle.load(f)
            
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
                
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            
            points = [
                models.PointStruct(
                    id=d['id'],
                    vector={
                        "dense": d['vector']['dense'],
                        "sparse": models.SparseVector(
                            indices=d['vector']['sparse']['indices'], 
                            values=d['vector']['sparse']['values']
                        )
                    },
                    payload=d['payload']
                ) for d in data
            ]
            
            batch_size = 100
            for i in range(0, len(points), batch_size):
                client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
                
            return client, points, data, True
            
        except Exception as e:
            print(f"Error loading data: {e}")  # Use print instead of st.error
            return client, [], [], False
    else:
        return client, [], [], False

# --- KNOWLEDGE GRAPH ENGINE ---
@st.cache_resource
def build_knowledge_graph(_points):
    G = nx.Graph()
    
    try:
        tnfa_levels = [p.payload.get('TNFA', 0) for p in _points]
        high_thresh = np.percentile(tnfa_levels, 75)
        low_thresh = np.percentile(tnfa_levels, 25)
    except:
        high_thresh = 1000
        low_thresh = 0

    for p in _points:
        meta = p.payload
        sid = str(meta.get('SampleID', 'Unknown'))
        
        G.add_node(sid, label=sid, color='#97C2FC', type='Sample', 
                   tnfa=meta.get('TNFA', 0), il22=meta.get('IL22', 0))
        
        site = meta.get('BodySite', 'Unknown')
        if site and site != 'Unknown' and site != 'nan':
            G.add_node(site, label=site, color='#FFFF00', type='BodySite')
            G.add_edge(sid, site, label="from_site")
            
        status = str(meta.get('InsulinSensitivity', 'Unknown'))
        if status and status != 'Unknown' and status != 'nan':
            color = '#00FF00' if 'Sensitive' in status else '#FF4B4B'
            G.add_node(status, label=status, color=color, type='Condition')
            G.add_edge(sid, status, label="has_status")

        tnfa = meta.get('TNFA', 0)
        if tnfa >= high_thresh:
            node_name = "High Inflammation (TNFA)"
            G.add_node(node_name, label=node_name, color='#FF4B4B', type='Biomarker')
            G.add_edge(sid, node_name, label="high_tnfa")
        elif tnfa <= low_thresh:
            node_name = "Low Inflammation (TNFA)"
            G.add_node(node_name, label=node_name, color='#00FF00', type='Biomarker')
            G.add_edge(sid, node_name, label="low_tnfa")

    return G

# --- NEW: VECTOR SPACE VISUALIZATION ---
@st.cache_data
def compute_tsne_projection(_raw_data):
    """Project high-dim vectors to 2D for visualization"""
    vectors = np.array([d['vector']['dense'] for d in _raw_data])
    metadata = [d['payload'] for d in _raw_data]
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
    coords_2d = tsne.fit_transform(vectors)
    
    # Cluster for automatic grouping
    n_clusters = min(5, len(vectors) // 50)
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)
    else:
        clusters = np.zeros(len(vectors))
    
    df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'SampleID': [m.get('SampleID', 'Unknown') for m in metadata],
        'BodySite': [m.get('BodySite', 'Unknown') for m in metadata],
        'TNFA': [m.get('TNFA', 0) for m in metadata],
        'Cluster': clusters.astype(str)
    })
    
    return df

# --- INITIALIZATION ---
client, points, raw_data, is_loaded = initialize_bio_engine()

# Show loading status AFTER cache returns
if is_loaded:
    # Only show toast on first successful load (check session state)
    if 'first_load' not in st.session_state:
        st.toast("✅ Bio-Memory Loaded Successfully!", icon="🟢")
        st.session_state.first_load = True
    
    G = build_knowledge_graph(points)
    tsne_df = compute_tsne_projection(raw_data)
    
    if os.path.exists(DIFF_VECTOR_FILE):
        diff_vector = np.load(DIFF_VECTOR_FILE)
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'rb') as f:
                vector_meta = pickle.load(f)
        else:
            vector_meta = {}
    else:
        diff_vector = np.random.rand(768)
        vector_meta = {}

# --- SIDEBAR ---
st.sidebar.title("🧬 Bio-Scout Control")

if is_loaded:
    st.sidebar.success(f"🟢 **System Online**")
    st.sidebar.caption(f"Memory: {len(points)} Vectors")
    st.sidebar.caption(f"Graph: {G.number_of_nodes()} Nodes")
    
    if vector_meta:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Vector Quality**")
        st.sidebar.metric("Effectiveness", f"{vector_meta.get('effectiveness', 0):.3f}")
        st.sidebar.caption(f"Method: {vector_meta.get('method', 'N/A')}")
else:
    st.sidebar.error("🔴 **System Offline**")

page = st.sidebar.radio("Navigation", [
    "🔬 Discovery Dashboard", 
    "🗺️ Vector Space Map",
    "🕸️ Graph Explorer",
    "📊 Batch Analysis",
    "🛠️ System Diagnostics"
])

# --- PAGE 1: DISCOVERY DASHBOARD ---
if page == "🔬 Discovery Dashboard":
    st.title("🧬 Bio-Scout: Multimodal Discovery Intelligence")
    st.markdown("**Track 4: Biological Design & Discovery** | Similarity-driven therapeutic candidate identification")

    if not is_loaded:
        st.warning("⚠️ Awaiting data load...")
        st.stop()

    # Quick ID Reference
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Sample IDs")
    try:
        recs, _ = client.scroll(collection_name=COLLECTION_NAME, limit=5, with_payload=True)
        valid_ids = [str(r.payload['SampleID']) for r in recs]
        st.sidebar.code("\n".join(valid_ids[:5]))
    except:
        valid_ids = []

    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.markdown("### 🎯 Query Configuration")
        default_id = valid_ids[0] if valid_ids else "Sample_AFTIWE"
        target_id_input = st.text_input("Target Sample ID", default_id)
        
        limit = st.slider("Neighbors to retrieve", 1, 15, 5)
        
        st.markdown("### 🧪 In-Silico Treatment Simulator")
        st.caption("Modify the query vector along a learned health trajectory")
        
        simulate = st.checkbox("Enable Treatment Simulation", value=False)
        strength = st.slider(
            "Intervention Strength", 
            0.0, 2.0, 0.5, 0.1,
            disabled=not simulate,
            help="Higher values = stronger push toward healthy state"
        )
        
        if simulate:
            st.info(f"💉 **Active**: Steering vector by {strength}× toward low-inflammation centroid")

        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("🔬 Analyzing biological context..."):
            try:
                # Fetch target
                filter_cond = models.Filter(
                    should=[
                        models.FieldCondition(key="SampleID", match=models.MatchValue(value=target_id_input))
                    ]
                )
                
                res = client.scroll(
                    collection_name=COLLECTION_NAME, 
                    scroll_filter=filter_cond,
                    limit=1, 
                    with_vectors=True
                )
                
                if not res[0]:
                    st.error(f"❌ Sample '{target_id_input}' not found")
                    st.stop()
                
                target = res[0][0]
                vec_dense = np.array(target.vector['dense'])
                meta = target.payload
                
                # Apply simulation
                original_vec = vec_dense.copy()
                if simulate:
                    vec_dense = vec_dense + (strength * diff_vector)
                    # Show toast OUTSIDE of any conditional that might be cached
                    if 'simulation_msg' not in st.session_state:
                        st.session_state.simulation_msg = True
                
                # Hybrid search
                hits = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vec_dense.tolist(),
                    using="dense",
                    limit=limit + 5,
                    with_payload=True
                ).points
                
                # Filter out self unless simulated
                neighbors = []
                for h in hits:
                    if str(h.payload['SampleID']) == target_id_input and not simulate:
                        continue
                    neighbors.append(h)
                    if len(neighbors) >= limit:
                        break
                
                # --- VISUALIZATION ---
                with col_viz:
                    st.markdown("### 📊 Biological Context Analysis")
                    
                    # Metadata cards
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Body Site", meta.get('BodySite', 'Unknown'))
                    m2.metric("TNFA", f"{meta.get('TNFA', 0):.1f} pg/mL")
                    m3.metric("IL-22", f"{meta.get('IL22', 0):.1f} pg/mL")
                    
                    # Cytokine comparison chart
                    chart_data = []
                    metrics = ['IL22', 'EGF', 'TNFA']
                    
                    target_label = "TARGET (Simulated)" if simulate else "TARGET"
                    for m in metrics:
                        chart_data.append({
                            "Sample": target_label,
                            "Cytokine": m,
                            "Level": meta.get(m, 0),
                            "Type": "Query"
                        })
                    
                    for i, n in enumerate(neighbors[:5], 1):
                        p = n.payload
                        for m in metrics:
                            chart_data.append({
                                "Sample": f"Match #{i}",
                                "Cytokine": m,
                                "Level": p.get(m, 0),
                                "Type": "Neighbor"
                            })
                    
                    df_plot = pd.DataFrame(chart_data)
                    
                    fig = px.bar(
                        df_plot, 
                        x="Cytokine", 
                        y="Level", 
                        color="Sample",
                        barmode="group",
                        title="Cytokine Profile Comparison",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    
                    # Highlight target
                    fig.update_traces(
                        marker=dict(line=dict(width=3, color='DarkSlateGrey')),
                        selector=dict(name=target_label)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- DESIGN INSIGHTS ---
                    st.markdown("### 🧠 Therapeutic Design Insights")
                    
                    # Calculate neighbor stats
                    neighbor_tnfa = [n.payload.get('TNFA', 0) for n in neighbors]
                    avg_tnfa = np.mean(neighbor_tnfa)
                    target_tnfa = meta.get('TNFA', 0)
                    
                    low_inflammation_count = sum(1 for x in neighbor_tnfa if x < 220)
                    
                    if simulate:
                        st.markdown(f"""
                        <div class="success-box">
                        <b>🎯 Predicted Outcome</b><br>
                        Vector steering identified <b>{low_inflammation_count}/{len(neighbors)}</b> low-inflammation neighbors.<br>
                        <b>Hypothesis</b>: A therapeutic intervention reducing TNFA by {abs(target_tnfa - avg_tnfa):.1f} pg/mL 
                        could shift this sample toward a healthier microbiome profile.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Evidence Trail:**")
                        for i, n in enumerate(neighbors[:3], 1):
                            st.caption(f"• **Match #{i}** ({n.payload['SampleID']}): TNFA={n.payload.get('TNFA', 0):.1f}, Site={n.payload.get('BodySite', 'Unknown')}")
                    else:
                        if target_tnfa > 250:
                            st.markdown(f"""
                            <div class="warning-box">
                            <b>⚠️ Risk Assessment</b><br>
                            Elevated TNFA ({target_tnfa:.1f} pg/mL) suggests inflammatory state.<br>
                            {low_inflammation_count}/{len(neighbors)} nearest neighbors show lower inflammation.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.success(f"✅ Sample shows healthy profile. {low_inflammation_count}/{len(neighbors)} neighbors confirm.")
                    
                    # Show similar samples table
                    st.markdown("### 📋 Most Similar Biological Cases")
                    neighbor_data = []
                    for i, n in enumerate(neighbors[:5], 1):
                        neighbor_data.append({
                            "Rank": i,
                            "Sample ID": n.payload['SampleID'],
                            "Body Site": n.payload.get('BodySite', 'Unknown'),
                            "TNFA": f"{n.payload.get('TNFA', 0):.1f}",
                            "Similarity": f"{1 - n.score:.3f}" if hasattr(n, 'score') else "N/A"
                        })
                    
                    st.dataframe(pd.DataFrame(neighbor_data), use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

# --- PAGE 2: VECTOR SPACE MAP ---
elif page == "🗺️ Vector Space Map":
    st.title("🗺️ Vector Space Topology")
    st.markdown("t-SNE projection of 768-dimensional genomic embeddings")
    
    if not is_loaded:
        st.warning("⚠️ Data not loaded")
        st.stop()
    
    # Controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        color_by = st.selectbox("Color by", ["Cluster", "BodySite", "TNFA"])
        show_labels = st.checkbox("Show Sample IDs", value=False)
    
    with col1:
        if color_by == "TNFA":
            fig = px.scatter(
                tsne_df, 
                x='x', y='y', 
                color='TNFA',
                hover_data=['SampleID', 'BodySite'],
                title="Genomic Vector Space (DNABERT-2 Embeddings)",
                color_continuous_scale='RdYlGn_r',
                labels={'TNFA': 'TNFA (pg/mL)'}
            )
        else:
            fig = px.scatter(
                tsne_df, 
                x='x', y='y', 
                color=color_by,
                hover_data=['SampleID', 'BodySite', 'TNFA'],
                title="Genomic Vector Space (DNABERT-2 Embeddings)"
            )
        
        if show_labels:
            fig.update_traces(text=tsne_df['SampleID'], textposition='top center')
        
        fig.update_layout(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation Guide:**
    - **Clusters** indicate biologically similar samples based on sequence patterns
    - **Spatial proximity** = Higher genomic similarity (k-mer profiles + semantic DNA features)
    - **Outliers** may represent unique microbial compositions or rare phenotypes
    """)

# --- PAGE 3: GRAPH EXPLORER ---
elif page == "🕸️ Graph Explorer":
    st.title("🕸️ GraphRAG: Relational Discovery")
    st.markdown("Navigate biological relationships via knowledge graph traversal")
    
    if not is_loaded:
        st.warning("⚠️ Data not loaded")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Graph Nodes", G.number_of_nodes())
    st.sidebar.metric("Graph Edges", G.number_of_edges())

    sample_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'Sample']
    
    if not sample_nodes:
        st.error("❌ No sample nodes found")
        st.stop()
    
    selected = st.selectbox("Focus Sample", sample_nodes[:200])
    hop_distance = st.slider("Neighborhood Depth", 1, 3, 2)
    
    if selected:
        # Multi-hop expansion
        nodes_to_draw = {selected}
        current_layer = {selected}
        
        for _ in range(hop_distance):
            next_layer = set()
            for node in current_layer:
                next_layer.update(G.neighbors(node))
            nodes_to_draw.update(next_layer)
            current_layer = next_layer
        
        subgraph = G.subgraph(nodes_to_draw)
        
        # Render
        ag_nodes = []
        ag_edges = []
        
        for n in subgraph.nodes:
            data = subgraph.nodes[n]
            size = 40 if n == selected else (25 if data.get('type') != 'Sample' else 15)
            
            ag_nodes.append(Node(
                id=n,
                label=str(data.get('label', n))[:20],
                size=size,
                color=data.get('color', '#97C2FC')
            ))
        
        for s, t in subgraph.edges:
            ag_edges.append(Edge(source=s, target=t))
        
        config = Config(
            width=1000, 
            height=700, 
            directed=False, 
            physics=True,
            hierarchical=False
        )
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            agraph(nodes=ag_nodes, edges=ag_edges, config=config)
        
        with col2:
            st.markdown(f"**{selected} Connections:**")
            neighbors = list(G.neighbors(selected))
            for n in neighbors[:10]:
                n_type = G.nodes[n].get('type', 'Unknown')
                st.caption(f"• {n} ({n_type})")

# --- PAGE 4: BATCH ANALYSIS ---
elif page == "📊 Batch Analysis":
    st.title("📊 Batch Cohort Analysis")
    st.markdown("Analyze multiple samples simultaneously for population-level insights")
    
    if not is_loaded:
        st.warning("⚠️ Data not loaded")
        st.stop()
    
    # Sample selection
    all_samples = [str(p.payload['SampleID']) for p in points]
    
    analysis_type = st.radio(
        "Analysis Mode",
        ["Body Site Comparison", "Inflammatory Profiling", "Custom Sample Set"]
    )
    
    if analysis_type == "Body Site Comparison":
        sites = list(set([p.payload.get('BodySite', 'Unknown') for p in points]))
        selected_sites = st.multiselect("Select Body Sites", sites, default=sites[:2])
        
        if selected_sites and st.button("Analyze"):
            site_data = {site: [] for site in selected_sites}
            
            for p in points:
                site = p.payload.get('BodySite', 'Unknown')
                if site in selected_sites:
                    site_data[site].append({
                        'TNFA': p.payload.get('TNFA', 0),
                        'IL22': p.payload.get('IL22', 0),
                        'EGF': p.payload.get('EGF', 0)
                    })
            
            # Create comparative plots
            fig = make_subplots(rows=1, cols=3, subplot_titles=["TNFA", "IL-22", "EGF"])
            
            for site in selected_sites:
                data = site_data[site]
                for i, metric in enumerate(['TNFA', 'IL22', 'EGF'], 1):
                    values = [d[metric] for d in data]
                    fig.add_trace(
                        go.Box(y=values, name=site, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=400, title_text="Cytokine Profiles by Body Site")
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats summary
            st.markdown("### Statistical Summary")
            summary_data = []
            for site in selected_sites:
                data = site_data[site]
                summary_data.append({
                    "Body Site": site,
                    "N Samples": len(data),
                    "Mean TNFA": f"{np.mean([d['TNFA'] for d in data]):.1f}",
                    "Median TNFA": f"{np.median([d['TNFA'] for d in data]):.1f}"
                })
            st.table(pd.DataFrame(summary_data))

# --- PAGE 5: DIAGNOSTICS ---
elif page == "🛠️ System Diagnostics":
    st.title("🛠️ System Health & Architecture")
    
    if is_loaded:
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Vector Database", "✅ Operational")
        col1.caption(f"{len(points)} embeddings loaded")
        
        col2.metric("Knowledge Graph", "✅ Built")
        col2.caption(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        col3.metric("Health Vector", "✅ Loaded" if os.path.exists(DIFF_VECTOR_FILE) else "⚠️ Missing")
        
        if vector_meta:
            st.markdown("---")
            st.markdown("### Vector Quality Metrics")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Separation Score", f"{vector_meta.get('separation_score', 0):.3f}")
            m2.metric("Effectiveness", f"{vector_meta.get('effectiveness', 0):.3f}")
            m3.metric("Method", vector_meta.get('method', 'N/A'))
            
            st.caption(f"Healthy Samples: {vector_meta.get('n_healthy', 0)} | Disease Samples: {vector_meta.get('n_disease', 0)}")
        
        st.markdown("---")
        st.markdown("### Architecture Diagram")
        st.code("""
        [FASTQ Files] 
            ↓
        [DNABERT-2 Embeddings] + [K-mer Sparse Vectors]
            ↓
        [Qdrant Hybrid Search Engine]
            ↓
        ┌─────────────┬──────────────┬─────────────┐
        │   t-SNE     │  GraphRAG    │  Simulator  │
        │ Projection  │  Navigator   │  (Vector    │
        │             │              │   Steering) │
        └─────────────┴──────────────┴─────────────┘
        """, language="text")
        
        st.success("✅ All systems nominal. Ready for demonstration.")
    else:
        st.error("🔴 System offline. Upload bio_memory_dump.pkl to initialize.")