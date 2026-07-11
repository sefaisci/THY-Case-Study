"""Regenerate the public architecture diagrams from maintained application code."""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.runnables.graph import CurveStyle, NodeStyles
from langchain_core.runnables.graph_png import PngDrawer
import pygraphviz as pgv

from model.agentic_rag.graphs import build_rag_graph
from model.agentic_rag.settings import RagSettings


ARCHITECTURE_DIR = PROJECT_ROOT / "docs" / "architecture"


LANGGRAPH_NODE_LABELS = {
    "__start__": "START",
    "query_understanding": "Understand + rewrite query\nBuild exactly 5 variants",
    "retrieval_subgraph:retrieval_planner": "Plan retrieval scope",
    "retrieval_subgraph:retrieve_variant": "Retrieve one query variant\n5 dynamic Send workers",
    "retrieval_subgraph:fuse_variant_results": "Fuse variant results\nDeterministic weighted RRF",
    "retrieval_subgraph:reranking": "Rerank evidence",
    "answer_subgraph:answer_generation": "Generate grounded answer",
    "answer_subgraph:citation_validation": "Validate exact chunk citations",
    "answer_subgraph:claim_evidence_reflection": "Evaluate claim grounding",
    "conversation_generation": "Generate conversational fallback\nActive-session history only",
    "final_response": "Build typed final response",
    "__end__": "END",
}

# LangGraph intentionally omits branch names from the expanded drawable graph
# when conditional functions return node names. Keeping the route descriptions
# here makes the public image self-explanatory; _assert_langgraph_shape below
# prevents these annotations from silently drifting away from executable code.
LANGGRAPH_EDGE_LABELS: dict[tuple[str, str], str | None] = {
    ("__start__", "query_understanding"): None,
    ("query_understanding", "retrieval_subgraph:retrieval_planner"): (
        "document retrieval"
    ),
    ("query_understanding", "conversation_generation"): (
        "history request / no documents"
    ),
    (
        "retrieval_subgraph:retrieval_planner",
        "retrieval_subgraph:retrieve_variant",
    ): "Send x5 query variants",
    (
        "retrieval_subgraph:retrieve_variant",
        "retrieval_subgraph:fuse_variant_results",
    ): "fan-in",
    (
        "retrieval_subgraph:fuse_variant_results",
        "retrieval_subgraph:reranking",
    ): None,
    (
        "retrieval_subgraph:reranking",
        "answer_subgraph:answer_generation",
    ): "sufficient evidence",
    ("retrieval_subgraph:reranking", "conversation_generation"): (
        "insufficient evidence"
    ),
    (
        "answer_subgraph:answer_generation",
        "answer_subgraph:citation_validation",
    ): None,
    (
        "answer_subgraph:citation_validation",
        "answer_subgraph:claim_evidence_reflection",
    ): None,
    (
        "answer_subgraph:claim_evidence_reflection",
        "final_response",
    ): "accepted + grounded",
    (
        "answer_subgraph:claim_evidence_reflection",
        "conversation_generation",
    ): "rejected / no-answer",
    ("conversation_generation", "final_response"): None,
    ("final_response", "__end__"): None,
}

LANGGRAPH_CONDITIONAL_EDGES = {
    ("query_understanding", "retrieval_subgraph:retrieval_planner"),
    ("query_understanding", "conversation_generation"),
    (
        "retrieval_subgraph:retrieval_planner",
        "retrieval_subgraph:retrieve_variant",
    ),
    (
        "retrieval_subgraph:reranking",
        "answer_subgraph:answer_generation",
    ),
    ("retrieval_subgraph:reranking", "conversation_generation"),
    (
        "answer_subgraph:claim_evidence_reflection",
        "final_response",
    ),
    (
        "answer_subgraph:claim_evidence_reflection",
        "conversation_generation",
    ),
}


class BrandPngDrawer(PngDrawer):
    """Render a LangGraph graph locally with restrained THY-inspired styling."""

    def get_node_label(self, label: str) -> str:
        """Return a safe multiline Graphviz HTML label."""

        label = self.labels.get("nodes", {}).get(label, label)
        lines = "<BR/>".join(html.escape(line) for line in label.splitlines())
        return f"<<B>{lines}</B>>"

    def get_edge_label(self, label: str) -> str:
        """Return a safe route label without hyperlink-like underlining."""

        label = self.labels.get("edges", {}).get(label, label)
        return f"<{html.escape(label)}>"

    def add_node(self, viz: Any, node: str) -> None:
        """Add one labeled graph node using the public repository palette."""

        fillcolor = "#FFFFFF"
        color = "#C8102E"
        if node.startswith("retrieval_subgraph:"):
            fillcolor = "#FFF7F7"
        elif node.startswith("answer_subgraph:"):
            fillcolor = "#F7FAFC"
            color = "#475569"
        elif node == "conversation_generation":
            fillcolor = "#FFF8E7"
            color = "#B45309"

        viz.add_node(
            node,
            label=self.get_node_label(node),
            shape="box",
            style="rounded,filled",
            fillcolor=fillcolor,
            color=color,
            fontcolor="#14213D",
            penwidth=1.4,
            fontsize=11,
            fontname=self.fontname,
            margin="0.16,0.10",
            group="fallback" if node == "conversation_generation" else "primary",
        )

    def add_edge(
        self,
        viz: Any,
        source: str,
        target: str,
        label: str | None = None,
        conditional: bool = False,
    ) -> None:
        """Render direct and conditional transitions with documented route names."""

        route_label = LANGGRAPH_EDGE_LABELS.get((source, target), label)
        viz.add_edge(
            source,
            target,
            label=self.get_edge_label(route_label) if route_label else "",
            color="#C8102E" if conditional else "#64748B",
            fontcolor="#475569",
            fontsize=9,
            fontname=self.fontname,
            penwidth=1.2,
            arrowsize=0.7,
            style="dashed" if conditional else "solid",
        )

    @staticmethod
    def update_styles(viz: Any, graph: Any) -> None:
        """Highlight the true LangGraph entry and terminal nodes."""

        if first := graph.first_node():
            viz.get_node(first.id).attr.update(
                fillcolor="#C8102E", color="#C8102E", fontcolor="#FFFFFF"
            )
        if last := graph.last_node():
            viz.get_node(last.id).attr.update(
                fillcolor="#14213D", color="#14213D", fontcolor="#FFFFFF"
            )

    def draw(self, graph: Any, output_path: str | None = None) -> bytes | None:
        """Use LangGraph's PNG drawer operations and save through local Graphviz."""

        viz = pgv.AGraph(
            directed=True,
            strict=False,
            rankdir="TB",
            bgcolor="white",
            nodesep=0.42,
            ranksep=0.72,
            pad=0.3,
            dpi=160,
            newrank=True,
            compound=True,
            splines="polyline",
            label=(
                "THY Cabin Knowledge Assistant\n"
                "LangGraph Agentic RAG - expanded executable graph"
            ),
            labelloc="t",
            labeljust="c",
            fontname=self.fontname,
            fontcolor="#14213D",
            fontsize=18,
        )
        self.add_nodes(viz, graph)
        self.add_edges(viz, graph)
        self.add_subgraph(viz, [node.split(":") for node in graph.nodes])
        for subgraph in viz.subgraphs():
            label = subgraph.name.replace("cluster_", "").replace("_", " ").title()
            subgraph.graph_attr.update(
                label=label,
                color="#CBD5E1",
                fontcolor="#475569",
                fontname=self.fontname,
                fontsize=10,
                labeljust="l",
                style="rounded",
                penwidth=1.0,
                margin=16,
            )
        self.update_styles(viz, graph)
        try:
            return cast(bytes | None, viz.draw(output_path, format="png", prog="dot"))
        finally:
            viz.close()

SYSTEM_ARCHITECTURE_DOT = r"""
digraph thy_agentic_rag_system {
  graph [
    rankdir=TB,
    newrank=true,
    compound=true,
    bgcolor="white",
    fontname="Helvetica",
    fontsize=22,
    label="THY-Branded Agentic RAG — Fully Asynchronous System Architecture",
    labelloc="t",
    labeljust="l",
    pad=0.35,
    nodesep=0.48,
    ranksep=0.95,
    splines=spline
  ];
  node [
    shape=box,
    style="rounded,filled",
    fontname="Helvetica",
    fontsize=11,
    color="#CBD5E1",
    fillcolor="#FFFFFF",
    fontcolor="#14213D",
    penwidth=1.2,
    margin="0.15,0.10"
  ];
  edge [
    fontname="Helvetica",
    fontsize=9,
    color="#64748B",
    fontcolor="#475569",
    arrowsize=0.7,
    penwidth=1.1
  ];

  user [label="Browser user", shape=oval, fillcolor="#FFF5F5", color="#C8102E"];

  subgraph cluster_frontend {
    label="Frontend";
    color="#E2E8F0";
    style="rounded";
    react [label="React 19 + TypeScript\nTailwind + TanStack Query + Zustand\nAsync mutations + batch job polling", fillcolor="#FFF5F5", color="#C8102E"];
    nginx [label="Nginx\nStatic delivery + async API proxy"];
    streamlit [label="Optional Streamlit\nCompatibility interface", style="rounded,dashed,filled", fillcolor="#F8FAFC"];
    react -> nginx;
  }

  subgraph cluster_backend {
    label="Async FastAPI application";
    color="#E2E8F0";
    style="rounded";
    api [label="Async REST API + OpenAPI\nPydantic v2 + request IDs", fillcolor="#F8FAFC"];
    identity [label="Tenant boundary\nUsername → internal user UUID", fillcolor="#FFF5F5", color="#C8102E"];
    document_service [label="Documents + ingestion\nConcurrent jobs + one-active-job invariant\nCancellation-safe lifecycle"];
    chat_service [label="Chat sessions\nPer-session ordered async turns\nBounded short-term history"];
    usage_service [label="Usage + pricing\nTokens, known cost, version"];
    repositories [label="Owner-scoped repositories\nAsync SQLAlchemy 2 + AsyncSession"];
    api -> identity;
    identity -> document_service;
    identity -> chat_service;
    identity -> usage_service;
    document_service -> repositories;
    chat_service -> repositories;
    usage_service -> repositories;
  }

  subgraph cluster_model {
    label="Model and orchestration layer";
    color="#E2E8F0";
    style="rounded";
    ingestion_pipeline [label="Bounded async ingestion\nConcurrent documents + semantic pages\nSemantic or Docling chunking\nBatched embeddings + Qdrant upserts", fillcolor="#FFF5F5", color="#C8102E"];
    rag_graph [label="Async LangGraph Agentic RAG\nGenerate exactly 5 query variants\nSend map → parallel retrieval → reducer\nRerank → answer → validate → reflect", fillcolor="#FFF5F5", color="#C8102E"];
    fallback [label="Conversational fallback\nActive-session history only"];
    rag_graph -> fallback [style=dashed];
  }

  subgraph cluster_data {
    label="Persistence";
    color="#E2E8F0";
    style="rounded";
    postgres [label="PostgreSQL 16\nAsync driver, row locks, partial unique index\nUsers, documents, jobs, chats, usage", shape=cylinder, fillcolor="#F8FAFC"];
    uploads [label="Persistent volume\nSources + render artifacts", shape=folder, fillcolor="#F8FAFC"];
    qdrant [label="External Qdrant\nAsyncQdrantClient\n5 parallel owner-scoped searches\nsemantic_chunks + docling_fixed_chunks", shape=cylinder, fillcolor="#F8FAFC"];
  }

  subgraph cluster_provider {
    label="External model provider";
    color="#E2E8F0";
    style="rounded";
    openai [label="OpenAI APIs via AsyncOpenAI\nResponses + embeddings", fillcolor="#F8FAFC"];
  }

  user -> react [label="HTTPS"];
  user -> streamlit [label="optional"];
  nginx -> api [label="async REST + X-Username"];
  streamlit -> api [label="REST + X-Username"];
  document_service -> ingestion_pipeline [label="bounded concurrent jobs"];
  chat_service -> rag_graph [label="await question + session history"];
  repositories -> postgres [label="await transactions"];
  document_service -> uploads [label="source files"];
  ingestion_pipeline -> qdrant [label="await batched owner-scoped upserts"];
  ingestion_pipeline -> openai [label="await Responses + embeddings"];
  rag_graph -> qdrant [label="5 parallel searches\nmandatory owner filter"];
  rag_graph -> openai [label="await rewrite + answer"];
  usage_service -> openai [style=dashed];
}
"""


def _assert_langgraph_shape(graph: Any) -> None:
    """Fail generation when the documented graph topology no longer matches code."""

    actual_nodes = set(graph.nodes)
    expected_nodes = set(LANGGRAPH_NODE_LABELS)
    if actual_nodes != expected_nodes:
        raise RuntimeError(
            "LangGraph nodes changed; update diagram labels and README documentation. "
            f"Missing labels: {sorted(actual_nodes - expected_nodes)}; "
            f"stale labels: {sorted(expected_nodes - actual_nodes)}."
        )

    actual_edges = {(edge.source, edge.target) for edge in graph.edges}
    expected_edges = set(LANGGRAPH_EDGE_LABELS)
    if actual_edges != expected_edges:
        raise RuntimeError(
            "LangGraph edges changed; update route annotations and README documentation. "
            f"Unannotated edges: {sorted(actual_edges - expected_edges)}; "
            f"stale annotations: {sorted(expected_edges - actual_edges)}."
        )

    actual_conditional_edges = {
        (edge.source, edge.target) for edge in graph.edges if edge.conditional
    }
    if actual_conditional_edges != LANGGRAPH_CONDITIONAL_EDGES:
        raise RuntimeError(
            "LangGraph conditional routing changed; update the diagram annotations. "
            f"Actual: {sorted(actual_conditional_edges)}."
        )


def generate_langgraph_diagram() -> None:
    """Use LangGraph's graph renderer to save the expanded Agentic RAG graph."""

    compiled_graph = build_rag_graph(settings=RagSettings())
    drawable_graph = compiled_graph.get_graph(xray=True)
    _assert_langgraph_shape(drawable_graph)
    styles = NodeStyles(
        default=(
            "fill:#FFF5F5,stroke:#C8102E,stroke-width:1.5px,"
            "color:#14213D,line-height:1.25"
        ),
        first="fill:#C8102E,stroke:#C8102E,color:#FFFFFF",
        last="fill:#14213D,stroke:#14213D,color:#FFFFFF",
    )
    mermaid_path = ARCHITECTURE_DIR / "langgraph-agentic-rag.mmd"
    png_path = ARCHITECTURE_DIR / "langgraph-agentic-rag.png"
    mermaid_path.write_text(
        drawable_graph.draw_mermaid(
            with_styles=True,
            curve_style=CurveStyle.LINEAR,
            node_colors=styles,
            wrap_label_n_words=4,
        ),
        encoding="utf-8",
    )
    BrandPngDrawer(
        fontname="DejaVu Sans",
        labels={
            "nodes": LANGGRAPH_NODE_LABELS,
            "edges": {},
        },
    ).draw(drawable_graph, str(png_path))
    png_bytes = png_path.read_bytes()
    if not png_bytes.startswith(b"\x89PNG"):
        raise RuntimeError("LangGraph did not return a valid PNG image.")


def generate_system_diagram() -> None:
    """Save the complete-system diagram from a versioned Graphviz source."""

    dot_executable = shutil.which("dot")
    if dot_executable is None:
        raise RuntimeError("Graphviz 'dot' is required to regenerate the system diagram.")
    dot_path = ARCHITECTURE_DIR / "complete-system-architecture.dot"
    png_path = ARCHITECTURE_DIR / "complete-system-architecture.png"
    svg_path = ARCHITECTURE_DIR / "complete-system-architecture.svg"
    dot_path.write_text(SYSTEM_ARCHITECTURE_DOT.strip() + "\n", encoding="utf-8")
    subprocess.run(
        [dot_executable, "-Tpng", "-Gdpi=180", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    subprocess.run(
        [dot_executable, "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )


def main() -> None:
    """Generate every architecture artifact used by the public README."""

    ARCHITECTURE_DIR.mkdir(parents=True, exist_ok=True)
    generate_langgraph_diagram()
    generate_system_diagram()
    print(f"Architecture diagrams written to {ARCHITECTURE_DIR}")


if __name__ == "__main__":
    main()
