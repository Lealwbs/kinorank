"""Editorial, minimalist Gradio application for KinoRank (Standalone & Hugging Face Spaces ready)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.agent.telemetry import format_telemetry_badge
from src.agent.tools import KinoRankAgentOrchestrator
from src.config import settings
from src.data.letterboxd import LetterboxdCatalogMatcher, parse_letterboxd_payload
from src.schemas.api import RecommendResponse
from src.schemas.domain import RankedRecommendation

EDITORIAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}

.kr-header {
    border-bottom: 1px solid #27272a;
    padding: 18px 4px 14px 4px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 12px;
}

.kr-brand {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f4f4f5;
}

.kr-brand span {
    color: #f59e0b;
    font-weight: 600;
}

.kr-subtitle {
    font-size: 0.86rem;
    color: #a1a1aa;
    font-family: 'JetBrains Mono', monospace;
}

.kr-telemetry-bar {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    color: #d4d4d8;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.kr-telemetry-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 8px;
}

.kr-profile-box {
    background: #121215;
    border-left: 3px solid #f59e0b;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 0.87rem;
    color: #d4d4d8;
    line-height: 1.45;
}

.kr-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 16px;
    margin-top: 8px;
}

.kr-card {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s ease;
}

.kr-card:hover {
    border-color: #52525b;
}

.kr-poster-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 3;
    background: #09090b;
    overflow: hidden;
}

.kr-poster-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.kr-rank-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(9, 9, 11, 0.88);
    border: 1px solid #3f3f46;
    color: #f4f4f5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
}

.kr-score-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(245, 158, 11, 0.92);
    color: #09090b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 3px 7px;
    border-radius: 4px;
}

.kr-card-body {
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    gap: 6px;
}

.kr-title {
    font-size: 0.96rem;
    font-weight: 600;
    color: #f4f4f5;
    line-height: 1.25;
}

.kr-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #a1a1aa;
}

.kr-genres {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 2px;
}

.kr-genre-pill {
    background: #27272a;
    color: #d4d4d8;
    font-size: 0.69rem;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
}

.kr-explanation {
    margin-top: auto;
    padding-top: 8px;
    border-top: 1px solid #27272a;
    font-size: 0.80rem;
    color: #d4d4d8;
    line-height: 1.38;
}
"""

SAMPLE_LETTERBOXD_CSV = """Date,Name,Year,Letterboxd URI,Rating
2026-10-01,Se7en,1995,https://boxd.it/29B4,5.0
2026-10-04,Fight Club,1999,https://boxd.it/2a9q,5.0
2026-10-08,The Matrix,1999,https://boxd.it/2a1m,4.5
2026-10-12,Memento,2000,https://boxd.it/2b5k,4.5
2026-10-15,Blade Runner,1982,https://boxd.it/298W,5.0
2026-10-18,Pulp Fiction,1994,https://boxd.it/29Pq,4.5
2026-10-22,Inception,2010,https://boxd.it/1skk,4.5
2026-10-28,The Silence of the Lambs,1991,https://boxd.it/297A,5.0
"""


def render_recommendations_html(response: RecommendResponse) -> str:
    """Render editorial HTML cards with TMDB posters, metadata, and telemetry bar."""
    t_str = html.escape(format_telemetry_badge(response.telemetry))
    prof = response.user_profile
    prof_summary = html.escape(prof.profile_summary_text if prof else "")
    agent_note = html.escape(response.agent_summary or "")

    cards_html: list[str] = []
    for item in response.recommendations:
        safe_title = html.escape(item.title)
        yr_str = str(item.release_year) if item.release_year else "Classic"
        dir_str = (
            f" · Dir. {html.escape(item.director)}"
            if item.director and item.director != "Unknown"
            else ""
        )
        meta_line = f"{yr_str} · {item.runtime_minutes}m{dir_str}"
        poster = html.escape(item.poster_url)
        exp = html.escape(item.explanation)

        genre_pills = "".join(
            f"<span class='kr-genre-pill'>{html.escape(g)}</span>" for g in item.genres[:3]
        )

        card = f"""
        <div class="kr-card">
            <div class="kr-poster-wrap">
                <img src="{poster}" alt="{safe_title}" loading="lazy" />
                <div class="kr-rank-badge">#{item.rank}</div>
                <div class="kr-score-badge">★ {item.bayesian_rating:.2f}</div>
            </div>
            <div class="kr-card-body">
                <div class="kr-title">{safe_title}</div>
                <div class="kr-meta">{meta_line}</div>
                <div class="kr-genres">{genre_pills}</div>
                <div class="kr-explanation">{exp}</div>
            </div>
        </div>
        """
        cards_html.append(card)

    grid_block = (
        f"<div class='kr-grid'>{''.join(cards_html)}</div>"
        if cards_html
        else "<div class='kr-profile-box'>No candidates matched all strict constraints.</div>"
    )

    return f"""
    <div>
        <div class="kr-telemetry-bar">
            <div><span class="kr-telemetry-dot"></span>{t_str}</div>
            <div>Candidates: {response.telemetry.candidates_retrieved} → {response.telemetry.final_returned}</div>
        </div>
        <div class="kr-profile-box">
            <strong>User Profile Context:</strong> {prof_summary}<br/>
            <strong>Stage 3 Governance:</strong> {agent_note}
        </div>
        {grid_block}
    </div>
    """


def build_shap_dataframe(recommendations: list[RankedRecommendation]) -> pd.DataFrame:
    """Convert RankedRecommendation feature contributions into an inspectable DataFrame."""
    rows: list[dict[str, Any]] = []
    for rec in recommendations:
        top_contribs = ", ".join(
            f"{k}: {v:+.3f}" for k, v in list(rec.feature_contributions.items())[:5]
        )
        rows.append(
            {
                "Rank": rec.rank,
                "Movie ID": rec.movie_id,
                "Title": f"{rec.title} ({rec.release_year or 'N/A'})",
                "RRF Score (Stage 1)": round(rec.rrf_score, 5),
                "LightGBM LambdaRank": round(rec.lgbm_score, 4),
                "MMR Score (Stage 2)": round(rec.mmr_score, 4),
                "Top TreeSHAP Feature Contributions": top_contribs,
            }
        )
    return pd.DataFrame(rows)


def create_gradio_interface(orchestrator: KinoRankAgentOrchestrator | None = None) -> gr.Blocks:
    """Create the 4-tab Editorial Gradio Blocks application."""
    orch = orchestrator or KinoRankAgentOrchestrator()
    matcher = LetterboxdCatalogMatcher(pd.DataFrame(list(orch.feature_store.items_by_id.values())))

    # Prepare top 250 popular movie choices for interactive Cold-Start selector
    seed_choices: list[tuple[str, int]] = []
    for mid in orch.feature_store.popular_movie_ids[:250]:
        item = orch.feature_store.get_item_dict(mid)
        if item:
            yr = item.get("release_year")
            label = f"{item.get('clean_title') or item.get('title')} ({int(yr) if pd.notna(yr) and yr else 'N/A'})"
            seed_choices.append((label, int(mid)))

    def handle_personalized_search(
        user_id: int,
        query: str,
        top_k: int,
        mmr_lambda: float,
        enable_agent: bool,
    ) -> tuple[str, pd.DataFrame, str, str]:
        resp = orch.recommend(
            user_id=int(user_id),
            query=query.strip() if query else None,
            top_k=int(top_k),
            mmr_lambda=float(mmr_lambda),
            enable_agentic_rerank=bool(enable_agent),
        )
        html_out = render_recommendations_html(resp)
        shap_df = build_shap_dataframe(resp.recommendations)
        intent_json = (
            json.dumps(resp.parsed_intent.model_dump(), indent=2, ensure_ascii=False)
            if resp.parsed_intent
            else "{}"
        )
        telemetry_json = json.dumps(resp.telemetry.model_dump(), indent=2, ensure_ascii=False)
        return html_out, shap_df, intent_json, telemetry_json

    def handle_cold_start(
        selected_seed_ids: list[int],
        nl_query: str,
        top_k: int,
        mmr_lambda: float,
    ) -> tuple[str, pd.DataFrame]:
        seeds = [int(x) for x in (selected_seed_ids or [])]
        if not seeds and not (nl_query and nl_query.strip()):
            seeds = [2959, 47, 2571]  # Default anchor: Fight Club, Se7en, The Matrix
        cold_prof = orch.feature_store.synthesize_cold_start_profile(seeds)
        resp = orch.recommend(
            user_id=None,
            query=nl_query.strip() if nl_query else None,
            seed_movie_ids=seeds,
            custom_user_profile=cold_prof,
            top_k=int(top_k),
            mmr_lambda=float(mmr_lambda),
            enable_agentic_rerank=True,
        )
        return render_recommendations_html(resp), build_shap_dataframe(resp.recommendations)

    def handle_letterboxd_import(
        uploaded_file: Any,
        use_sample: bool,
        lb_query: str,
        top_k: int,
        mmr_lambda: float,
    ) -> tuple[str, str]:
        if uploaded_file is not None and not use_sample:
            file_path = Path(
                uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            )
            raw_bytes = file_path.read_bytes()
            fname = file_path.name
        else:
            raw_bytes = SAMPLE_LETTERBOXD_CSV.encode("utf-8")
            fname = "sample_letterboxd_ratings.csv"

        import_res, lb_prof = parse_letterboxd_payload(
            file_bytes=raw_bytes,
            filename=fname,
            matcher=matcher,
        )
        resp = orch.recommend(
            user_id=None,
            query=lb_query.strip() if lb_query else None,
            seed_movie_ids=lb_prof.watched_movie_ids[-15:],
            custom_user_profile=lb_prof,
            top_k=int(top_k),
            mmr_lambda=float(mmr_lambda),
            enable_agentic_rerank=True,
        )
        summary_json = json.dumps(import_res.model_dump(), indent=2, ensure_ascii=False)
        return render_recommendations_html(resp), summary_json

    benchmark_md_path = settings.metrics_dir / "benchmark_results.md"
    benchmark_png_path = settings.metrics_dir / "benchmark_comparison.png"
    benchmark_md_content = (
        benchmark_md_path.read_text(encoding="utf-8")
        if benchmark_md_path.exists()
        else "Run `python -m src.evaluation.benchmark` to generate benchmark metrics."
    )

    with gr.Blocks(title="KinoRank") as demo:
        gr.HTML(
            f"<style>{EDITORIAL_CSS}</style>"
            """
            <div class="kr-header">
                <div>
                    <div class="kr-brand">KINO<span>RANK</span></div>
                    <div class="kr-subtitle">ALS + FAISS RRF · LightGBM LambdaRank · LLM Reranker</div>
                </div>
                <div class="kr-subtitle">Letterboxd Compatible</div>
            </div>
            """
        )

        with gr.Tabs():
            # TAB 1: Busca
            with gr.TabItem("Busca"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_id_input = gr.Number(
                            label="MovieLens User ID (1 - 610)",
                            value=1,
                            minimum=1,
                            maximum=610,
                            precision=0,
                        )
                        query_input = gr.Textbox(
                            label="Conversational Intent / Constraints (PT or EN)",
                            placeholder="Ex: thriller psicológico estilo Fincher, < 135 min, depois de 1994, sem terror",
                            lines=2,
                        )
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                label="Top-K Recommendations",
                                minimum=4,
                                maximum=20,
                                step=1,
                                value=8,
                            )
                            mmr_slider = gr.Slider(
                                label="MMR Relevance vs. Diversity (λ)",
                                minimum=0.4,
                                maximum=1.0,
                                step=0.05,
                                value=0.75,
                            )
                        agent_toggle = gr.Checkbox(
                            label="Enable Stage 3 Agentic Rerank & Grounded Explanations",
                            value=True,
                        )
                        run_btn = gr.Button("Generate Recommendations", variant="primary")

                        gr.Examples(
                            examples=[
                                [
                                    1,
                                    "thriller psicológico estilo Fincher, < 135 min, depois de 1994, sem terror",
                                    8,
                                    0.75,
                                    True,
                                ],
                                [
                                    15,
                                    "mind-bending sci-fi like Nolan or Villeneuve after 2000 without romance",
                                    8,
                                    0.70,
                                    True,
                                ],
                                [
                                    42,
                                    "crime drama or heist movie from the 90s under 160 minutes",
                                    8,
                                    0.80,
                                    True,
                                ],
                                [
                                    105,
                                    "animação japonesa estilo Miyazaki ou fantasia familiar",
                                    8,
                                    0.75,
                                    True,
                                ],
                            ],
                            inputs=[
                                user_id_input,
                                query_input,
                                top_k_slider,
                                mmr_slider,
                                agent_toggle,
                            ],
                            label="Preset Conversational Scenarios",
                        )

                    with gr.Column(scale=3):
                        recs_html_output = gr.HTML(label="Recommendations")

            # TAB 2: Cold-Start
            with gr.TabItem("Cold-Start"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_dropdown = gr.Dropdown(
                            choices=seed_choices,
                            multiselect=True,
                            max_choices=6,
                            value=[2959, 47, 2571],
                            label="Escolha 3 a 5 filmes favoritos",
                        )
                        cold_query = gr.Textbox(
                            label="Preferencia em texto livre (opcional)",
                            placeholder="Ex: dark neo-noir mystery with twist endings under 130 minutes",
                            lines=2,
                        )
                        cold_topk = gr.Slider(label="Top-K", minimum=4, maximum=16, step=1, value=8)
                        cold_mmr = gr.Slider(
                            label="MMR Diversity (λ)",
                            minimum=0.4,
                            maximum=1.0,
                            step=0.05,
                            value=0.75,
                        )
                        cold_btn = gr.Button("Recomendar", variant="primary")
                    with gr.Column(scale=3):
                        cold_html_output = gr.HTML()
                        cold_shap_table = gr.Dataframe(label="Pesos TreeSHAP")

            # TAB 3: Letterboxd
            with gr.TabItem("Letterboxd"):
                with gr.Row():
                    with gr.Column(scale=1):
                        lb_file = gr.File(
                            label="Arquivo Letterboxd (.zip, ratings.csv, watched.csv, watchlist.csv)",
                            file_types=[".csv", ".zip"],
                        )
                        lb_use_sample = gr.Checkbox(
                            label="Usar perfil de exemplo (8 filmes avaliados)",
                            value=True,
                        )
                        lb_query_box = gr.Textbox(
                            label="Filtro adicional (opcional)",
                            placeholder="Ex: sci-fi or thriller after 1995 under 140 min",
                        )
                        lb_topk = gr.Slider(label="Top-K", minimum=4, maximum=16, step=1, value=8)
                        lb_mmr = gr.Slider(
                            label="MMR Diversity (λ)",
                            minimum=0.4,
                            maximum=1.0,
                            step=0.05,
                            value=0.75,
                        )
                        lb_btn = gr.Button("Importar", variant="primary")
                    with gr.Column(scale=3):
                        lb_html_output = gr.HTML()
                        lb_summary_json = gr.Code(
                            label="Resumo da importacao",
                            language="json",
                        )

            # TAB 4: Metricas
            with gr.TabItem("Métricas"):
                gr.Markdown("### Pesos")
                shap_output_table = gr.Dataframe(label="Contribuicao por feature (TreeSHAP)")
                with gr.Row():
                    intent_json_box = gr.Code(label="Filtros extraidos (Pydantic)", language="json")
                    telemetry_json_box = gr.Code(label="Telemetria (JSON)", language="json")
                gr.Markdown("---")
                gr.Markdown(benchmark_md_content)
                if benchmark_png_path.exists():
                    gr.Image(
                        value=str(benchmark_png_path),
                        label="Comparativo offline (split temporal)",
                    )

        # Wire events
        run_btn.click(
            fn=handle_personalized_search,
            inputs=[user_id_input, query_input, top_k_slider, mmr_slider, agent_toggle],
            outputs=[recs_html_output, shap_output_table, intent_json_box, telemetry_json_box],
        )
        cold_btn.click(
            fn=handle_cold_start,
            inputs=[seed_dropdown, cold_query, cold_topk, cold_mmr],
            outputs=[cold_html_output, cold_shap_table],
        )
        lb_btn.click(
            fn=handle_letterboxd_import,
            inputs=[lb_file, lb_use_sample, lb_query_box, lb_topk, lb_mmr],
            outputs=[lb_html_output, lb_summary_json],
        )

        # Populate initial default state on load
        demo.load(
            fn=handle_personalized_search,
            inputs=[user_id_input, query_input, top_k_slider, mmr_slider, agent_toggle],
            outputs=[recs_html_output, shap_output_table, intent_json_box, telemetry_json_box],
        )

    return demo


if __name__ == "__main__":
    app_ui = create_gradio_interface()
    app_ui.launch(server_name="0.0.0.0", server_port=7860)
