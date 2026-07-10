"""Regenerate the public architecture diagrams from maintained application code."""

from __future__ import annotations

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


class BrandPngDrawer(PngDrawer):
    """Render a LangGraph graph locally with restrained THY-inspired styling."""

    def add_node(self, viz: Any, node: str) -> None:
        """Add one labeled graph node using the public repository palette."""

        viz.add_node(
            node,
            label=self.get_node_label(node),
            shape="box",
            style="rounded,filled",
            fillcolor="#FFF7F7",
            color="#C8102E",
            fontcolor="#14213D",
            penwidth=1.4,
            fontsize=12,
            fontname=self.fontname,
            margin="0.16,0.10",
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
            rankdir="LR",
            bgcolor="white",
            nodesep=0.45,
            ranksep=0.7,
            pad=0.25,
            splines="spline",
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
                style="rounded",
                penwidth=1.0,
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
    label="THY-Branded Agentic RAG — Complete System Architecture",
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
    react [label="React 19 + TypeScript\nTailwind + TanStack Query + Zustand", fillcolor="#FFF5F5", color="#C8102E"];
    nginx [label="Nginx\nStatic delivery + API proxy"];
    streamlit [label="Optional Streamlit\nCompatibility interface", style="rounded,dashed,filled", fillcolor="#F8FAFC"];
    react -> nginx;
  }

  subgraph cluster_backend {
    label="FastAPI application";
    color="#E2E8F0";
    style="rounded";
    api [label="REST API + OpenAPI\nPydantic v2 + request IDs", fillcolor="#F8FAFC"];
    identity [label="Tenant boundary\nUsername → internal user UUID", fillcolor="#FFF5F5", color="#C8102E"];
    document_service [label="Documents + ingestion\nValidation and lifecycle"];
    chat_service [label="Chat sessions\nBounded short-term history"];
    usage_service [label="Usage + pricing\nTokens, known cost, version"];
    repositories [label="Owner-scoped repositories\nSQLAlchemy 2"];
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
    ingestion_pipeline [label="Multi-format ingestion\nPage-image rendering\nSemantic or Docling chunking", fillcolor="#FFF5F5", color="#C8102E"];
    rag_graph [label="LangGraph Agentic RAG\nPlan → retrieve → rerank → answer\nValidate → reflect → finalize", fillcolor="#FFF5F5", color="#C8102E"];
    fallback [label="Conversational fallback\nActive-session history only"];
    rag_graph -> fallback [style=dashed];
  }

  subgraph cluster_data {
    label="Persistence";
    color="#E2E8F0";
    style="rounded";
    postgres [label="PostgreSQL 16\nUsers, documents, jobs, chats, usage", shape=cylinder, fillcolor="#F8FAFC"];
    uploads [label="Persistent volume\nSources + render artifacts", shape=folder, fillcolor="#F8FAFC"];
    qdrant [label="External Qdrant\nsemantic_chunks\ndocling_fixed_chunks", shape=cylinder, fillcolor="#F8FAFC"];
  }

  subgraph cluster_provider {
    label="External model provider";
    color="#E2E8F0";
    style="rounded";
    openai [label="OpenAI APIs\nResponses + embeddings", fillcolor="#F8FAFC"];
  }

  user -> react [label="HTTPS"];
  user -> streamlit [label="optional"];
  nginx -> api [label="REST + X-Username"];
  streamlit -> api [label="REST + X-Username"];
  document_service -> ingestion_pipeline [label="ingestion job"];
  chat_service -> rag_graph [label="question + session history"];
  repositories -> postgres [label="transactions"];
  document_service -> uploads [label="source files"];
  ingestion_pipeline -> qdrant [label="owner-scoped chunks"];
  ingestion_pipeline -> openai;
  rag_graph -> qdrant [label="mandatory owner filter"];
  rag_graph -> openai;
  usage_service -> openai [style=dashed];
}
"""


def generate_langgraph_diagram() -> None:
    """Use LangGraph's graph renderer to save the expanded Agentic RAG graph."""

    compiled_graph = build_rag_graph(settings=RagSettings())
    drawable_graph = compiled_graph.get_graph(xray=True)
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
            "nodes": {
                "__start__": "START",
                "query_understanding": "Understand query",
                "retrieval_subgraph:retrieval_planner": "Plan retrieval",
                "retrieval_subgraph:hybrid_retrieval": "Hybrid retrieval",
                "retrieval_subgraph:reranking": "Rerank evidence",
                "answer_subgraph:answer_generation": "Generate grounded answer",
                "answer_subgraph:citation_validation": "Validate citations",
                "answer_subgraph:claim_evidence_reflection": "Reflect on evidence",
                "conversation_generation": "Conversational fallback",
                "final_response": "Build final response",
                "__end__": "END",
            },
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
