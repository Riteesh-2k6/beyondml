#!/usr/bin/env python3
"""
BeyondML — AI Agent Orchestration Platform
Terminal-native AutoML · Groq LLM · Genetic Algorithm

Run: conda run -n beyondml python tui_app.py
Quit: Ctrl+C or Escape
"""

import sys
import os
import asyncio

# Add project root to path so beyondml package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

import pandas as pd
import numpy as np
from textual.app import App, ComposeResult
from textual.screen import Screen, ModalScreen
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Tree, RichLog, Static, DataTable,
    Sparkline, Input, Button, RadioSet, RadioButton, Label, Rule,
    ProgressBar,
)
from textual.binding import Binding
from textual.message import Message
from textual import work
from rich.text import Text

from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
from beyondml.agents.orchestrator import OrchestratorAgent
from beyondml.agents.eda_agent import EDAAgent
from beyondml.agents.outlier_agent import OutlierAgent
from beyondml.agents.feature_agent import FeatureAgent
from beyondml.agents.ga_trainer import GATrainerAgent
from beyondml.agents.evaluator_agent import EvaluatorAgent
from beyondml.agents.reflection_agent import ReflectionAgent
from beyondml.llm import get_llm_provider
from beyondml.engine.tracing import AgentTrace


# ═══════════════════════════════════════════════════
#  ASCII Art Banner
# ═══════════════════════════════════════════════════

BANNER = """[bold orange3]
  ██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗ ███╗   ███╗██╗     
  ██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗████╗ ████║██║     
  ██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║██╔████╔██║██║     
  ██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║██║╚██╔╝██║██║     
  ██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝██║ ╚═╝ ██║███████╗
  ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝     ╚═╝╚══════╝[/bold orange3]"""

SUBTITLE = "[dim]Terminal-native AutoML · Ollama / Groq · Genetic Algorithm · Ctrl+C to quit[/dim]"


# ═══════════════════════════════════════════════════
#  Completion Modal
# ═══════════════════════════════════════════════════

class CompletionModal(ModalScreen):
    """Shows final pipeline results."""

    CSS = """
    CompletionModal {
        align: center middle;
    }
    CompletionModal > Vertical {
        width: 80;
        max-height: 30;
        border: heavy $accent;
        background: $surface;
        padding: 2;
    }
    CompletionModal .modal-title {
        text-align: center;
        text-style: bold;
        color: $success;
        margin-bottom: 1;
    }
    CompletionModal .close-btn {
        margin-top: 1;
        width: 100%;
    }
    """

    def __init__(self, results: dict):
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        r = self.results
        with Vertical():
            yield Static("🎉  Pipeline Complete!", classes="modal-title")
            yield Rule()
            yield Static(f"\n[bold green]Test Score:[/bold green] {r.get('test_score', 'N/A')}")
            yield Static(f"\n[bold]Best Hyperparameters:[/bold]")
            params = r.get("best_params", {})
            for k, v in params.items():
                yield Static(f"    {k}: {v}")
            model_path = r.get("model_path", "N/A")
            yield Static(f"\n[bold]Model saved:[/bold]\n    {model_path}")
            yield Rule()
            yield Button("Close  [ctrl+q]", variant="warning", classes="close-btn", id="close-modal")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "close-modal":
            self.dismiss()

    def key_ctrl_q(self):
        self.dismiss()


# ═══════════════════════════════════════════════════
#  Welcome Screen
# ═══════════════════════════════════════════════════

class WelcomeScreen(Screen):
    """Compact boot screen — all fields visible on a 24-row terminal."""

    CSS = """
    WelcomeScreen {
        background: $background;
        layout: vertical;
        overflow-y: auto;
    }
    .banner { text-align: center; }
    .subtitle { text-align: center; }
    .field-label { color: $accent; margin: 0; padding: 0; }
    #bottom-bar {
        dock: bottom;
        height: auto;
        padding: 0 2;
        background: $surface;
        border-top: solid $accent;
    }
    .start-btn { width: 100%; margin: 0; }
    #loading-bar { display: none; }
    #loading-bar.visible { display: block; }
    #loading-status { text-align: center; display: none; color: $warning; }
    #loading-status.visible { display: block; }
    #path-info { text-align: center; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Static(BANNER, classes="banner", markup=True)
        yield Static(SUBTITLE, classes="subtitle", markup=True)
        yield Rule()
        yield Label("[bold]> CSV file path[/bold]  [dim](default: data/echallan_daily_data.csv)[/dim]", classes="field-label")
        yield Input(value="data/echallan_daily_data.csv", id="csv-path", placeholder="data/sample.csv")
        yield Label("[bold]> Dataset description[/bold]  [dim](optional)[/dim]", classes="field-label")
        yield Input(id="description", placeholder="e.g. Iris flower measurements for species classification")
        yield Label("[bold]> Analysis path[/bold]", classes="field-label")
        with RadioSet(id="path-select"):
            yield RadioButton("Autonomous  (Let AI decide)", value=True)
            yield RadioButton("Explore  (EDA only)")
            yield RadioButton("Supervised ML  (classification / regression)")
            yield RadioButton("Unsupervised ML  (clustering)")
        yield Label("[bold]> LLM Provider[/bold]", classes="field-label")
        with RadioSet(id="llm-select"):
            yield RadioButton("Ollama  (Local, private)", value=True)
            yield RadioButton("Groq  (Cloud, fast)")
        with Horizontal():
            with Vertical():
                yield Label("[bold]> Pop Size[/bold]  [dim](GA)[/dim]", classes="field-label")
                yield Input(value="10", id="ga-pop", placeholder="10")
            with Vertical():
                yield Label("[bold]> Generations[/bold]  [dim](GA)[/dim]", classes="field-label")
                yield Input(value="5", id="ga-gen", placeholder="5")
        # Bottom bar — always visible
        with Vertical(id="bottom-bar"):
            yield Static("[dim]Path: Explore (EDA only)[/dim]", id="path-info")
            yield Button("▸ Start Pipeline", variant="warning", classes="start-btn", id="start-btn")
            yield ProgressBar(id="loading-bar", show_eta=False, show_percentage=False)
            yield Static("[bold yellow]⏳ Loading dataset...[/bold yellow]", id="loading-status")

    def on_radio_set_changed(self, event: RadioSet.Changed):
        """Update the path info label when user selects a radio."""
        labels = {
            0: "Autonomous (Let AI decide)", 
            1: "Explore Dataset (EDA only)", 
            2: "Supervised ML (classification / regression)", 
            3: "Unsupervised ML (clustering)"
        }
        idx = event.radio_set.pressed_index
        name = labels.get(idx, "Autonomous")
        self.query_one("#path-info", Static).update(f"[dim]Path: [bold]{name}[/bold][/dim]")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "start-btn":
            self._start_pipeline()

    @work(thread=False)
    async def _start_pipeline(self):
        btn = self.query_one("#start-btn", Button)
        bar = self.query_one("#loading-bar", ProgressBar)
        status = self.query_one("#loading-status", Static)

        # Show loading state
        btn.disabled = True
        btn.label = "⏳ Loading..."
        bar.add_class("visible")
        status.add_class("visible")

        csv_path = self.query_one("#csv-path", Input).value.strip()
        description = self.query_one("#description", Input).value.strip()
        radio = self.query_one("#path-select", RadioSet)
        idx = radio.pressed_index
        path_map = {0: "autonomous", 1: "explore", 2: "supervised", 3: "unsupervised"}
        path_choice = path_map.get(idx, "autonomous")

        llm_radio = self.query_one("#llm-select", RadioSet)
        llm_idx = llm_radio.pressed_index
        llm_choice = "ollama" if llm_idx == 0 else "groq"

        ga_pop = int(self.query_one("#ga-pop", Input).value.strip() or "10")
        ga_gen = int(self.query_one("#ga-gen", Input).value.strip() or "5")

        # Resolve CSV path
        full_path = csv_path
        if not os.path.isabs(csv_path):
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

        if not os.path.exists(full_path):
            self.notify(f"File not found: {full_path}", severity="error")
            self._reset_button()
            return

        status.update("[bold yellow]⏳ Reading CSV file...[/bold yellow]")
        await asyncio.sleep(0.1)

        try:
            ext = os.path.splitext(full_path)[1].lower()
            if ext in ['.xlsx', '.xls']:
                df = await asyncio.to_thread(pd.read_excel, full_path)
            elif ext == '.json':
                df = await asyncio.to_thread(pd.read_json, full_path)
            else:
                df = await asyncio.to_thread(pd.read_csv, full_path)
            
            # Scikit-learn fails on inf values; scrub them globally to NaN
            df = df.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.notify(f"Error loading CSV: {e}", severity="error")
            self._reset_button()
            return

        status.update(f"[bold green]✓ Loaded {df.shape[0]} rows × {df.shape[1]} cols — launching pipeline...[/bold green]")
        await asyncio.sleep(0.5)

        self.app.push_screen(
            PipelineScreen(df, full_path, description, path_choice, llm_choice, ga_pop, ga_gen)
        )

    def _reset_button(self):
        """Reset button to original state on error."""
        btn = self.query_one("#start-btn", Button)
        bar = self.query_one("#loading-bar", ProgressBar)
        status = self.query_one("#loading-status", Static)
        btn.disabled = False
        btn.label = "▸ Start Pipeline"
        bar.remove_class("visible")
        status.remove_class("visible")


# ═══════════════════════════════════════════════════
#  Pipeline Screen
# ═══════════════════════════════════════════════════

class PipelineScreen(Screen):
    """Main 3-column pipeline execution screen."""

    CSS = """
    PipelineScreen {
        layout: grid;
        grid-size: 3 2;
        grid-columns: 1fr 3fr 1fr;
        grid-rows: 1fr auto;
    }

    #left-panel {
        row-span: 1;
        border: solid $accent;
        height: 100%;
    }
    #pipeline-tree {
        height: auto;
        max-height: 40%;
        scrollbar-size: 0 0;
    }
    #reasoning-log {
        height: 1fr;
        border-top: tall $accent;
        scrollbar-size: 0 0;
        background: $boost;
    }
    #center-panel {
        row-span: 1;
        border: solid $warning;
        height: 100%;
        scrollbar-size: 0 0;
    }
    #right-panel {
        row-span: 1;
        border: solid $accent;
        height: 100%;
        scrollbar-size: 0 0;
    }
    RichLog {
        scrollbar-size: 0 0;
    }
    Tree {
        scrollbar-size: 0 0;
    }
    #input-bar {
        column-span: 3;
        height: auto;
        min-height: 5;
        max-height: 8;
        border: solid $accent;
        background: $surface;
        padding: 0 1;
    }

    .panel-title {
        background: $surface;
        color: $text-muted;
        text-style: bold;
        padding: 0 1;
    }
    .stat-label { color: $text-muted; }
    .stat-value { color: $text; text-style: bold; }
    .section-header {
        color: $text-muted;
        margin: 1 0 0 0;
        text-style: bold;
    }
    #input-prompt { color: $warning; }
    #user-input { display: none; }
    #user-input.visible { display: block; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("escape", "quit", "Quit"),
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("ctrl+e", "export", "Export", show=True),
        Binding("tab", "focus_next", "Focus", show=True),
    ]

    def __init__(self, df: pd.DataFrame, path: str, description: str, path_choice: str, llm_choice: str = "ollama", ga_pop: int = 10, ga_gen: int = 5):
        super().__init__()
        self.df = df
        self.dataset_path = path
        self.description = description
        self.path_choice = path_choice
        self.llm_choice = llm_choice
        self.ga_pop = ga_pop
        self.ga_gen = ga_gen
        self.input_queue = asyncio.Queue()
        self._fitness_data = []

    def compose(self) -> ComposeResult:
        # Left: Pipeline Tree + Reasoning
        with Vertical(id="left-panel"):
            yield Static("▸ Pipeline · Tree", classes="panel-title")
            tree = Tree("Pipeline", id="pipeline-tree")
            tree.root.expand()
            yield tree
            yield Static("▸ Agent Reasoning", classes="panel-title")
            yield RichLog(id="reasoning-log", markup=True, wrap=True, max_lines=1000)

        # Center: Log + Charts
        with Vertical(id="center-panel"):
            yield Static("▸ Agent Log · RichLog + Charts", classes="panel-title")
            yield RichLog(id="main-log", markup=True, wrap=True, max_lines=5000)

        # Right: Stats
        with VerticalScroll(id="right-panel"):
            yield Static("▸ Stats · Data · GA Progr", classes="panel-title")
            yield Static("DATASET", classes="section-header")
            yield Static("shape: ×", id="stat-shape")
            yield Static("target: —", id="stat-target")
            yield Static("task: —", id="stat-task")
            yield Static("classes: —", id="stat-classes")
            yield Static("train/test: — / —", id="stat-split")
            yield Static("new features: 0", id="stat-features")
            yield Rule()
            yield Static("GA PROGRESS", classes="section-header")
            yield Sparkline([], id="fitness-sparkline", summary_function=max)
            yield Rule()
            yield Static("DATA PREVIEW", classes="section-header")
            yield DataTable(id="data-preview", zebra_stripes=True)

        # Bottom: Input bar — always visible
        with Vertical(id="input-bar"):
            yield Static("[dim]ctrl+c quit · ctrl+s save · ctrl+e export · tab focus[/dim]", id="keybinds-label")
            yield Static("", id="input-prompt")
            yield Input(id="user-input", placeholder="Waiting for agent prompt...")

    def on_mount(self) -> None:
        # Update initial stats
        self._update_stat("stat-shape", f"shape: {self.df.shape[0]} × {self.df.shape[1]}")

        # Populate data preview
        # Populate data preview fully
        table = self.query_one("#data-preview", DataTable)
        cols = list(self.df.columns)[:20]  # Show up to 20 columns
        table.add_columns(*[c[:15] for c in cols])
        for _, row in self.df.head(15).iterrows():
            table.add_row(*[str(row[c])[:15] for c in cols])

        # Start the pipeline
        self.run_pipeline()

    def _update_stat(self, widget_id: str, value: str):
        try:
            self.query_one(f"#{widget_id}", Static).update(value)
        except Exception:
            pass

    async def _log(self, msg: str):
        """Write to the center log panel."""
        try:
            log = self.query_one("#main-log", RichLog)
            log.write(msg)
        except Exception:
            pass

    async def _update_reasoning(self, agent_name: str, text: str):
        """Update the reasoning log in the left panel."""
        try:
            log = self.query_one("#reasoning-log", RichLog)
            log.write(f"\n[bold cyan]● {agent_name}[/bold cyan]")
            log.write(f"[dim]{text}[/dim]")
        except Exception:
            pass

    async def _get_user_input(self, prompt: str = "Your response...") -> str:
        """Show input bar with prompt, wait for user response, then hide."""
        inp = self.query_one("#user-input", Input)
        prompt_label = self.query_one("#input-prompt", Static)

        # Show the prompt and input field
        prompt_label.update(f"[bold yellow]▸ {prompt}[/bold yellow]")
        inp.placeholder = "Type your answer and press Enter..."
        inp.add_class("visible")
        inp.focus()

        # Also log the prompt so user can see it in the center panel
        await self._log(f"\n[bold yellow]⌨ INPUT NEEDED:[/bold yellow] {prompt}")

        # Wait for input submission
        response = await self.input_queue.get()

        # Hide and reset
        inp.remove_class("visible")
        inp.value = ""
        prompt_label.update("")

        await self._log(f"  [dim]User entered: {response or '(empty — using default)'}[/dim]")
        return response

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "user-input":
            self.input_queue.put_nowait(event.value)

    def _update_tree_node(self, tree: Tree, node_name: str, status: str, details: list = None):
        """Add or update a node in the pipeline tree."""
        icon = {"pending": "⬜", "running": "🔄", "done": "✅", "error": "❌"}.get(status, "⬜")
        node = tree.root.add(f"{icon} {node_name}")
        if details:
            for d in details:
                node.add_leaf(f"  {d}")
        node.expand()
        return node

    @work(thread=False)
    async def run_pipeline(self):
        """Main pipeline execution — runs all agents sequentially."""
        tree = self.query_one("#pipeline-tree", Tree)
        trace = AgentTrace()

        try:
            # Initialize LLM based on user selection
            await self._log(f"[dim]Initializing LLM provider ({self.llm_choice})...[/dim]")
            try:
                if self.llm_choice == "groq":
                    from beyondml.llm.groq_provider import GroqProvider
                    llm = GroqProvider()
                else:
                    from beyondml.llm.ollama_provider import OllamaProvider
                    llm = OllamaProvider()
                await self._log(f"[green]\u2713 Connected to {llm.model_name}[/green]\n")
            except Exception as e:
                await self._log(f"[bold red]\u2717 LLM init failed: {e}[/bold red]")
                if self.llm_choice == "groq":
                    await self._log("[yellow]Set GROQ_API_KEY in .env file[/yellow]")
                else:
                    await self._log("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
                return

            # ── STEP 1: Orchestrator ──
            await self._log("[dim]─── Orchestrator ─────────────────────────────────[/dim]")
            self._update_tree_node(tree, "Orchestrator", "running")
            trace.start("Orchestrator", f"path_choice={self.path_choice}")

            identifier = TargetIdentifier(self.df)
            target_info = identifier.identify()

            df_summary = (
                f"Shape: {self.df.shape}\n"
                f"Columns: {list(self.df.columns)}\n"
                f"Dtypes: {self.df.dtypes.to_dict()}\n"
                f"First 3 rows:\n{self.df.head(3).to_string()}\n"
                f"Describe:\n{self.df.describe().to_string()[:1000]}"
            )

            orch = OrchestratorAgent(llm)
            orch_result = await orch.run(
                df_summary, self.description, target_info, self.path_choice, self._log
            )

            path = orch_result.get("path", "supervised")
            target = orch_result.get("suggested_target", target_info.get("suggested_target"))
            model_recs = orch_result.get("model_recommendations", ["RandomForest"])

            self._update_tree_node(tree, "Orchestrator", "done", [f"Path: {path}"])
            trace.finish(f"path={path}, target={target}")
            await self._update_reasoning("Orchestrator", orch_result.get("reasoning", "Autonomous routing decided."))
            await self._log("")

            # ── STEP 2: EDA Agent ──
            await self._log("[dim]─── EDA Agent ───────────────────────────────────[/dim]")
            self._update_tree_node(tree, "EDA Agent", "running")
            trace.start("EDA Agent", f"shape={self.df.shape}")

            profiler = DatasetProfiler(self.df, target_column=target)
            profile = profiler.run()

            eda = EDAAgent(llm)
            eda_result = await eda.run(self.df, profile, target_info, self.description, self._log)

            # Render charts in log
            for chart_name, chart_str in eda_result.get("rendered_charts", []):
                await self._log(f"\n[bold magenta]── {chart_name} ──[/bold magenta]")
                try:
                    # Safely parse raw ANSI from plotext
                    ansi_chart = Text.from_ansi(chart_str, no_wrap=True)
                    await self._log(ansi_chart)
                except Exception:
                    await self._log("  [dim]⚠ Could not render chart output[/dim]")

            confirmed_target = eda_result.get("suggested_target") or target
            task_type = eda_result.get("task_type", "classification")

            # Update stats
            self._update_stat("stat-target", f"target: [bold green]{confirmed_target or '—'}[/bold green]")
            self._update_stat("stat-task", f"task: [bold]{task_type.title()}[/bold]")
            if profile.get("target_analysis"):
                nu = profile["target_analysis"].get("num_unique", "—")
                self._update_stat("stat-classes", f"classes: [bold]{nu}[/bold]")
            tr = int(self.df.shape[0] * 0.8)
            te = self.df.shape[0] - tr
            self._update_stat("stat-split", f"train/test: {tr} / {te}")

            self._update_tree_node(tree, "EDA Agent", "done",
                [f"Target: {confirmed_target}", f"{len(eda_result.get('eda_insights', []))} insights"])
            trace.finish(f"target={confirmed_target}, insights={len(eda_result.get('eda_insights', []))}")
            await self._update_reasoning("EDA Agent", eda_result.get("narrative", "Data profiling and chart generation complete."))
            await self._log("")

            # For explore-only path, stop here
            if path == "explore":
                await self._log("\n[bold green]✓ Exploration complete![/bold green]")
                self._update_tree_node(tree, "Export", "done")
                return

            # Re-profile with confirmed target
            if confirmed_target and confirmed_target != target:
                profiler = DatasetProfiler(self.df, target_column=confirmed_target)
                profile = profiler.run()

            # ── STEP 3: Outlier Handler ──
            await self._log("[dim]─── Outlier Handler ─────────────────────────────[/dim]")
            self._update_tree_node(tree, "Outlier Handler", "running")
            trace.start("Outlier Handler")

            outlier_agent = OutlierAgent(llm)
            outlier_result = await outlier_agent.run(
                self.df, profile.get("outlier_summary", {}), profile, self._log, self._get_user_input
            )
            self.df = outlier_result["df"]

            self._update_tree_node(tree, "Outlier Handler", "done",
                [f"Strategy: {outlier_result['outlier_strategy']}"])
            trace.finish(f"strategy={outlier_result['outlier_strategy']}")
            await self._update_reasoning("Outlier Handler", f"Applied {outlier_result['outlier_strategy']} strategy based on distribution analysis.")
            await self._log("")

            # --- START ITERATIVE LOOP ---
            max_iterations = 3
            current_iter = 1
            best_model_score = -float('inf')
            best_eval_result = None
            
            # This holds insights for FeatureEngineer. We'll append reflection feedback to it.
            current_insights = eda_result.get("eda_insights", [])
            
            # Identify model choice once
            model_choice = model_recs[0] if model_recs else "RandomForest"
            current_pop_size = self.ga_pop
            current_generations = self.ga_gen
            
            while current_iter <= max_iterations:
                if current_iter > 1:
                    await self._log(f"\n[bold magenta]─── OPTIMIZATION LOOP {current_iter}/{max_iterations} ───[/bold magenta]")
                    self._update_tree_node(tree, f"Iter {current_iter}", "running")
                
                # ── STEP 4: Feature Engineering ──
                await self._log("[dim]─── Feature Engineer ────────────────────────────[/dim]")
                self._update_tree_node(tree, "Feature Engineer", "running")

                # Re-profile after outlier handling or previous loops
                profiler = DatasetProfiler(self.df, target_column=confirmed_target)
                profile = profiler.run()

                feat_agent = FeatureAgent(llm)
                feat_result = await feat_agent.run(self.df, profile, current_insights, self._log)
                self.df = feat_result["df"]
                
                # Fix infs that might have been created by Feature Engineer mathematical expressions (div by zero)
                self.df = self.df.replace([np.inf, -np.inf], np.nan)

                n_applied = len(feat_result.get("features_applied", []))
                self._update_stat("stat-features", f"new features: [bold orange1]{n_applied}[/bold orange1]")
                self._update_stat("stat-shape", f"shape: {self.df.shape[0]} × {self.df.shape[1]}")

                self._update_tree_node(tree, "Feature Engineer", "done",
                    [f"+{n_applied} features"])
                
                # Combine rationales for reasoning log
                applied_list = feat_result.get("features_applied", [])
                feat_reasoning = "\n".join([f"• {f['name']}: {f['rationale']}" for f in feat_result.get("feature_proposals", []) if f['name'] in applied_list])
                await self._update_reasoning("Feature Engineer", feat_reasoning or "No new features derived.")
                await self._log("")

                # Update data preview
                try:
                    table = self.query_one("#data-preview", DataTable)
                    table.clear(columns=True)
                    cols = list(self.df.columns)[:20]
                    table.add_columns(*[c[:15] for c in cols])
                    for _, row in self.df.head(15).iterrows():
                        table.add_row(*[str(row[c])[:15] for c in cols])
                except Exception:
                    pass

                if path == "unsupervised":
                    # ── Unsupervised path ──
                    await self._log("[dim]─── Clustering Agent ────────────────────────────[/dim]")
                    self._update_tree_node(tree, "Clustering", "running")

                    from beyondml.engine.unsupervised import UnsupervisedPipeline
                    unsup = UnsupervisedPipeline(self.df, profile)
                    results = await asyncio.to_thread(unsup.run_clustering)

                    for task_name, metrics in results.items():
                        await self._log(f"\n  [bold cyan]{task_name}[/bold cyan]")
                        for k, v in metrics.items():
                            await self._log(f"    {k}: {v}")

                    self._update_tree_node(tree, "Clustering", "done")
                    await self._log("\n[bold green]✓ Unsupervised analysis complete![/bold green]")
                    return

                # ── STEP 5: GA Trainer (supervised) ──
                await self._log("[dim]─── GA Trainer ──────────────────────── [running] ──[/dim]")
                self._update_tree_node(tree, "GA Trainer", "running")

                # Re-profile with new features
                profiler = DatasetProfiler(self.df, target_column=confirmed_target)
                profile = profiler.run()

                async def on_ga_progress(gen_summary):
                    self._fitness_data.append(gen_summary["best_fitness"] * 100)
                    try:
                        sparkline = self.query_one("#fitness-sparkline", Sparkline)
                        sparkline.data = self._fitness_data.copy()
                    except Exception:
                        pass

                ga_agent = GATrainerAgent(llm)
                ga_result = await ga_agent.run(
                    df=self.df,
                    target_column=confirmed_target,
                    profile=profile,
                    model_choice=model_choice,
                    log=self._log,
                    get_user_input=self._get_user_input,
                    on_ga_progress=on_ga_progress,
                    pop_size=current_pop_size,
                    generations=current_generations,
                )

                self._update_tree_node(tree, "GA Trainer", "done",
                    [f"Best: {ga_result['best_cv_score']:.4f}", f"Model: {ga_result['model_type']}"])
                await self._log("")

                # ── STEP 6: Evaluator ──
                await self._log("[dim]─── Evaluator ───────────────────────────────────[/dim]")
                self._update_tree_node(tree, "Evaluator", "running")

                eval_agent = EvaluatorAgent(llm)
                eval_result = await eval_agent.run(
                    df=self.df,
                    target_column=confirmed_target,
                    profile=profile,
                    best_params=ga_result["best_params"],
                    model_type=ga_result["model_type"],
                    problem_type=profile["target_analysis"]["target_type"],
                    log=self._log,
                )

                self._update_tree_node(tree, "Evaluator", "done",
                    [f"Score: {eval_result['test_score']:.4f}"])
                await self._update_reasoning("Evaluator", eval_result.get("eval_narration", "Final model performance validated."))
                
                # ── STEP 7: Reflection ──
                from beyondml.agents.reflection_agent import ReflectionAgent
                await self._log("\n[dim]─── Reflection Agent ───────────────────────────[/dim]")
                self._update_tree_node(tree, f"Reflection", "running")
                
                reflection_agent = ReflectionAgent(llm)
                reflection_result = await reflection_agent.run(eval_result, current_iter, max_iterations, self._log)
                
                self._update_tree_node(tree, f"Reflection", "done", [reflection_result["status"]])
                await self._update_reasoning("Reflection", reflection_result.get("reasoning", "Pipeline iteration completed."))
                
                # Track best
                if eval_result['test_score'] > best_model_score:
                    best_model_score = eval_result['test_score']
                    best_eval_result = eval_result
                    
                if reflection_result["status"] in ("satisfied", "error"):
                    break
                    
                mods = reflection_result.get("modifications") or {}
                if mods:
                    current_insights.append({
                        "finding": f"Reflection Feedback: Must improve score. Rationale: {reflection_result.get('reasoning')} New Features requested: {mods.get('new_features')}",
                        "severity": "high"
                    })
                    # Process drops immediately
                    for drop_col in mods.get("features_to_drop", []):
                        # Strip scikit-learn pipeline prefixes like 'num__' or 'cat__'
                        clean_col = drop_col.split("__")[-1] if "__" in drop_col else drop_col
                        
                        # Also handle any extra whitespace or quoting LLM might have sent
                        clean_col = clean_col.strip().strip("'").strip('"')
                        
                        if clean_col in self.df.columns and clean_col != confirmed_target:
                            self.df = self.df.drop(columns=[clean_col])
                            await self._log(f"  [red]−[/red] [dim]Successfully Dropped '{clean_col}' from dataset.[/dim]")
                        else:
                            await self._log(f"  [yellow]⚠[/yellow] [dim]Could not find '{clean_col}' in dataset to drop.[/dim]")
                            
                    # Update hyperparameters and logic for the next iteration from AI Reflection dictation
                    if mods.get("next_model"):
                        model_choice = mods["next_model"]
                    if mods.get("next_ga_generations"):
                        current_generations = mods["next_ga_generations"]
                    if mods.get("next_ga_pop_size"):
                        current_pop_size = mods["next_ga_pop_size"]
                
                current_iter += 1

            # Log trace summary
            await self._log("\n[dim]─── Pipeline Trace ──────────────────────────────[/dim]")
            await self._log(f"[dim]{trace.print_summary()}[/dim]")

            # Show completion modal
            await asyncio.sleep(0.5)
            self.app.push_screen(CompletionModal(best_eval_result))

        except Exception as e:
            await self._log(f"\n[bold red]Pipeline error: {e}[/bold red]")
            import traceback
            await self._log(f"[dim]{traceback.format_exc()}[/dim]")

    def action_save(self):
        self.notify("State saved!", severity="information")

    def action_export(self):
        self.notify("Report exported!", severity="information")

    def action_quit(self):
        self.app.exit()


# ═══════════════════════════════════════════════════
#  Main App
# ═══════════════════════════════════════════════════

class BeyondMLApp(App):
    """BeyondML — AI Agent Orchestration Platform."""

    CSS = """
    Screen { background: #0a0c10; }
    """

    TITLE = "BeyondML"
    SUB_TITLE = "AI Agent Orchestration Platform"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
    ]

    def on_mount(self):
        self.push_screen(WelcomeScreen())


if __name__ == "__main__":
    app = BeyondMLApp()
    app.run()
