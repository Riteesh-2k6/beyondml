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

def save_env_config(config: dict):
    """Save configuration to .env file."""
    env_path = Path(__file__).parent / ".env"
    lines = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()
        keys_handled = set()
        for line in existing_lines:
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=", 1)[0].strip()
                if key in config:
                    lines.append(f"{key}={config[key]}")
                    keys_handled.add(key)
                else:
                    lines.append(line)
            else:
                lines.append(line)
        
        # Add any new keys
        for key, val in config.items():
            if key not in keys_handled:
                lines.append(f"{key}={val}")
    else:
        for key, val in config.items():
            lines.append(f"{key}={val}")
            
    env_path.write_text("\n".join(lines) + "\n")
    # Update current environment
    for key, val in config.items():
        os.environ[key] = str(val)


def check_config() -> bool:
    """Check if basic LLM configuration exists."""
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        return False
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        return False
    return True


# Load .env on startup
from pathlib import Path
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
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
from beyondml.agents import (
    OrchestratorAgent, EDAAgent, OutlierAgent, FeatureAgent, 
    GATrainerAgent, EvaluatorAgent, ReflectionAgent, SanityAgent, 
    LeakageAgent, ImputationAgent, DeepLearningAgent
)
from beyondml.llm import get_llm_provider
from beyondml.engine.tracing import AgentTrace


# ═══════════════════════════════════════════════════
#  ASCII Art Banner
# ═══════════════════════════════════════════════════

BANNER = """[bold orange3]
  ██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗    ███╗   ███╗██╗     
  ██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗   ████╗ ████║██║     
  ██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║   ██╔████╔██║██║     
  ██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║   ██║╚██╔╝██║██║     
  ██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝   ██║ ╚══ ██║███████╗
  ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝    ╚═╝     ╚═╝╚══════╝[/bold orange3]"""

SUBTITLE = "[dim]Terminal-native AutoML · Ollama / Groq · Genetic Algorithm · Ctrl+C to quit[/dim]"


# ═══════════════════════════════════════════════════
#  Completion Modal
# ═══════════════════════════════════════════════════

class ConfigScreen(ModalScreen):
    """Setup screen for LLM provider details."""

    CSS = """
    ConfigScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    ConfigScreen > Vertical {
        width: 60;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 2;
    }
    ConfigScreen .title {
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    ConfigScreen .label {
        margin-top: 1;
        color: $text-muted;
    }
    ConfigScreen .save-btn {
        margin-top: 2;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🤖 LLM SETUP WIZARD", classes="title")
            yield Static("Configure your LLM provider to get started.", classes="label")
            yield Rule()
            
            yield Label("LLM Provider")
            with RadioSet(id="setup-llm-select"):
                yield RadioButton("Groq (Cloud)", value=os.getenv("LLM_PROVIDER") == "groq" or not os.getenv("LLM_PROVIDER"))
                yield RadioButton("Ollama (Local)", value=os.getenv("LLM_PROVIDER") == "ollama")
            
            yield Label("Groq API Key (required for Groq)")
            yield Input(value=os.getenv("GROQ_API_KEY", ""), id="setup-groq-key", password=True, placeholder="gsk_...")
            
            yield Label("Groq Model")
            yield Input(value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), id="setup-groq-model")
            
            yield Label("Ollama Model")
            yield Input(value=os.getenv("OLLAMA_MODEL", "qwen3:8b"), id="setup-ollama-model")
            
            yield Button("Save & Continue", variant="success", classes="save-btn", id="save-config-btn")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save-config-btn":
            llm_radio = self.query_one("#setup-llm-select", RadioSet)
            provider = "groq" if llm_radio.pressed_index == 0 else "ollama"
            
            groq_key = self.query_one("#setup-groq-key", Input).value.strip()
            groq_model = self.query_one("#setup-groq-model", Input).value.strip()
            ollama_model = self.query_one("#setup-ollama-model", Input).value.strip()
            
            if provider == "groq" and not groq_key:
                self.notify("Groq API Key is required for Groq provider!", severity="error")
                return
                
            config = {
                "LLM_PROVIDER": provider,
                "GROQ_API_KEY": groq_key,
                "GROQ_MODEL": groq_model,
                "OLLAMA_MODEL": ollama_model
            }
            
            save_env_config(config)
            self.app.notify("Configuration saved to .env", severity="success")
            self.dismiss(True)


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
        with VerticalScroll():
            yield Static("🎉  Pipeline Complete!", classes="modal-title")
            yield Rule()
            yield Static(f"\n[bold green]Test Score:[/bold green] {r.get('test_score', 'N/A')}")
            yield Static(f"\n[bold]Best Hyperparameters:[/bold]")
            params = r.get("best_params", {})
            for k, v in params.items():
                yield Static(f"    {k}: {v}")
            model_path = r.get("model_path", "N/A")
            yield Static(f"\n[bold]Model saved:[/bold]\n    {model_path}")
            
            # Explainability
            xai = r.get("xai_result", {})
            if xai and xai.get("status") == "success":
                yield Rule()
                yield Static("[bold magenta]🔍 Explainability (SHAP)[/bold magenta]")
                yield Static(f"    [italic]\"{xai.get('explanation', '')}\"[/italic]\n")
                for d in xai.get("drivers", []):
                    yield Static(f"    • {d}")

            yield Rule()
            with Horizontal():
                yield Button("Export to Jupyter Notebook", variant="success", id="export-notebook-btn")
                yield Button("Close", variant="warning", id="close-modal")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "close-modal":
            self.dismiss()
        elif event.button.id == "export-notebook-btn":
            self.dismiss()
            if hasattr(self.app.screen, "action_export"):
                self.app.screen.action_export()

    def key_ctrl_q(self):
        self.dismiss()


# ═══════════════════════════════════════════════════
#  Welcome Screen
# ═══════════════════════════════════════════════════

class DatasetInput(Input):
    """Input field that cycles through default datasets with up/down keys."""
    
    BINDINGS = [
        Binding("up", "history_up", "Previous Dataset", show=False),
        Binding("down", "history_down", "Next Dataset", show=False),
    ]

    def on_mount(self):
        self.dataset_list = [
            "data/titanic.csv",
            "data/adult.csv",
            "data/breast_cancer.csv"
        ]
        # Append existing echallan data if it exists or just keep it in rotation
        self.dataset_list.append("data/echallan_daily_data.csv")
        
        self.idx = 0
        self.value = self.dataset_list[self.idx]

    def action_history_up(self):
        self.idx = (self.idx - 1) % len(self.dataset_list)
        self.value = self.dataset_list[self.idx]
        self.action_end()  # Move cursor to end

    def action_history_down(self):
        self.idx = (self.idx + 1) % len(self.dataset_list)
        self.value = self.dataset_list[self.idx]
        self.action_end()


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
    #ga-config { height: auto; min-height: 4; }
    """

    def compose(self) -> ComposeResult:
        yield Static(BANNER, classes="banner", markup=True)
        yield Static(SUBTITLE, classes="subtitle", markup=True)
        yield Rule()
        yield Label("[bold]> CSV file path[/bold]  [dim](Use Up/Down arrows to cycle defaults)[/dim]", classes="field-label")
        yield DatasetInput(id="csv-path", placeholder="data/sample.csv")
        yield Label("[bold]> Dataset description[/bold]  [dim](optional)[/dim]", classes="field-label")
        yield Input(id="description", placeholder="e.g. Iris flower measurements for species classification")
        yield Label("[bold]> Inference CSV (for Drift Analysis)[/bold]  [dim](optional)[/dim]", classes="field-label")
        yield Input(id="inference-path", placeholder="data/inference_sample.csv")
        yield Label("[bold]> Analysis path[/bold]", classes="field-label")
        with RadioSet(id="path-select"):
            yield RadioButton("Autonomous  (Let AI decide)", value=True)
            yield RadioButton("Explore  (EDA only)")
            yield RadioButton("Supervised ML  (classification / regression)")
            yield RadioButton("Unsupervised ML  (clustering)")
            yield RadioButton("Dimensionality Reduction  (PCA / feature extraction)")
            yield RadioButton("Deep Learning  (Neural Networks)")
        yield Label("[bold]> LLM Provider[/bold]", classes="field-label")
        with RadioSet(id="llm-select"):
            yield RadioButton("Groq  (Cloud, fast)", value=True)
            yield RadioButton("Ollama  (Local, private)")
        with Horizontal(id="ga-config", classes="config-row"):
            with Vertical():
                yield Label("[bold]> Pop Size[/bold]  [dim](GA)[/dim]", classes="field-label")
                yield Input(value="10", id="ga-pop", placeholder="10")
            with Vertical():
                yield Label("[bold]> Split Size[/bold]  [dim](Test %)[/dim]", classes="field-label")
                yield Input(value="0.20", id="test-size", placeholder="0.20")
            with Vertical():
                yield Label("[bold]> Generations[/bold]  [dim](GA)[/dim]", classes="field-label")
                yield Input(value="5", id="ga-gen", placeholder="5")
        yield Label("[bold]> Auto-Ensemble[/bold]  [dim](Combine top 3 models)[/dim]", classes="field-label")
        with RadioSet(id="ensemble-select"):
            yield RadioButton("No  (Single Best Model)", value=True)
            yield RadioButton("Yes  (Voting Ensemble)")
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
            3: "Unsupervised ML (clustering)",
            4: "Dimensionality Reduction (PCA)",
            5: "Deep Learning (Neural Networks)"
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
        path_map = {0: "autonomous", 1: "explore", 2: "supervised", 3: "unsupervised", 4: "dimensionality_reduction", 5: "deep_learning"}
        path_choice = path_map.get(idx, "autonomous")

        llm_radio = self.query_one("#llm-select", RadioSet)
        llm_idx = llm_radio.pressed_index
        llm_choice = "groq" if llm_idx == 0 else "ollama"

        ga_pop = int(self.query_one("#ga-pop", Input).value.strip() or "10")
        ga_gen = int(self.query_one("#ga-gen", Input).value.strip() or "5")
        test_size = float(self.query_one("#test-size", Input).value.strip() or "0.20")
        
        ensemble_radio = self.query_one("#ensemble-select", RadioSet)
        use_ensemble = ensemble_radio.pressed_index == 1

        # Resolve CSV path
        full_path = csv_path
        if not os.path.isabs(csv_path):
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

        if not os.path.exists(full_path):
            self.notify(f"File not found: {full_path}", severity="error")
            self._reset_button()
            return

        inference_csv_path = self.query_one("#inference-path", Input).value.strip()

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
            PipelineScreen(df, full_path, description, path_choice, llm_choice, ga_pop, ga_gen, inference_csv_path, target_col="", use_ensemble=use_ensemble, test_size=test_size)
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
        Binding("v", "view_charts", "View Charts", show=True),
        Binding("tab", "focus_next", "Focus", show=True),
    ]

    def __init__(self, df: pd.DataFrame, path: str, description: str, path_choice: str, llm_choice: str = "ollama", ga_pop: int = 10, ga_gen: int = 5, inference_path: str = "", target_col: str = "", use_ensemble: bool = False, test_size: float = 0.20):
        super().__init__()
        self.df = df
        self.inference_path = inference_path
        self.user_target = target_col
        self.dataset_path = path
        self.description = description
        self.path_choice = path_choice
        self.llm_choice = llm_choice
        self.ga_pop = ga_pop
        self.ga_gen = ga_gen
        self.use_ensemble = use_ensemble
        self.test_size = test_size
        self.input_queue = asyncio.Queue()
        self._fitness_data = []
        self.interactive_charts = []

    def compose(self) -> ComposeResult:
        # Left: Pipeline Tree + Reasoning
        with Vertical(id="left-panel"):
            yield Static("▸ Pipeline · Tree", classes="panel-title")
            tree = Tree("Pipeline", id="pipeline-tree")
            tree.root.expand()
            tree.root.add("Orchestrator")
            tree.root.add("Domain Sanity (Parallel)")
            tree.root.add("Data Drift Analysis")
            tree.root.add("Imputation")
            tree.root.add("EDA Agent (Parallel)")
            tree.root.add("Outlier Handler (Parallel)")
            tree.root.add("Feature Engineer")
            tree.root.add("GA Trainer (Supervised)")
            tree.root.add("Ensemble")
            tree.root.add("DL Trainer (Neural Net)")
            tree.root.add("Evaluator")
            tree.root.add("Explainability")
            tree.root.add("Reflection")
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
            yield Static("[dim]ctrl+c quit · ctrl+s save · ctrl+e export · v view charts · tab focus[/dim]", id="keybinds-label")
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
        """Write to the center log panel with timestamp."""
        try:
            from datetime import datetime
            now = datetime.now().strftime("%H:%M:%S")
            log = self.query_one("#main-log", RichLog)
            log.write(f"[dim]{now}[/dim] {msg}")
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
                
                # Verify connection
                await self._log(f"[dim]Testing connection to {llm.model_name}...[/dim]")
                if await asyncio.to_thread(llm.test_connection):
                    await self._log(f"[green]✓ Connected to {llm.model_name}[/green]\n")
                else:
                    await self._log(f"[bold red]✗ Connection failed to {llm.model_name}[/bold red]")
                    if "ollama" in self.llm_choice:
                        await self._log("[yellow]Tip: Ensure Ollama is running and you have run 'ollama pull " + (os.getenv("OLLAMA_MODEL", "qwen3:8b")) + "'[/yellow]\n")
                    else:
                        await self._log("[yellow]Tip: Check your GROQ_API_KEY in .env[/yellow]\n")
                    # We don't necessarily stop here, but the user is warned.
            except Exception as e:
                await self._log(f"[bold red]✗ LLM init failed: {e}[/bold red]")
                if self.llm_choice == "groq":
                    await self._log("[yellow]Set GROQ_API_KEY in .env file[/yellow]")
                else:
                    await self._log("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
                return

            # ── STEP 1: Orchestrator ──
            await self._log("[dim]─── Orchestrator ─────────────────────────────────[/dim]")
            self._update_tree_node(tree, "Orchestrator", "running")
            trace.start("Orchestrator", f"path_choice={self.path_choice}")

            # Sample values for LLM
            df_sample_str = ""
            for col in self.df.columns[:20]: # cap at 20 columns max to not overflow context
                uniques = self.df[col].dropna().unique()
                if len(uniques) > 0:
                    sample = [str(x) for x in uniques[:3]]
                    df_sample_str += f" - {col}: {sample}\n"

            df_summary = (
                f"Shape: {self.df.shape}\n"
                f"Columns: {list(self.df.columns)}\n"
                f"Dtypes: {self.df.dtypes.to_dict()}\n"
                f"Sample Values:\n{df_sample_str}\n"
                f"Describe:\n{self.df.describe().to_string()[:1000]}"
            )

            orch = OrchestratorAgent(llm)
            orch_result = await orch.run(
                df_summary, self.description, self.user_target, self.path_choice, self._log
            )

            path = orch_result.get("path", "supervised")
            target = orch_result.get("suggested_target")
            
            if path in ["supervised", "deep_learning"]:
                # Interactive prompt to confirm or change target
                cols_str = ", ".join(list(self.df.columns))
                user_override = await self._get_user_input(
                    f"Columns: {cols_str}\nAI suggests target '{target}'. Press Enter to accept, or type a different column name:"
                )
                if user_override and user_override.strip() in self.df.columns:
                    target = user_override.strip()
                    await self._log(f"[bold green]Target overridden to: {target}[/bold green]")
                elif user_override and user_override.strip() not in self.df.columns:
                    await self._log(f"[bold red]Column '{user_override}' not found. Defaulting to '{target}'.[/bold red]")
                else:
                    await self._log(f"[dim]Target confirmed as '{target}'.[/dim]")

                if self.user_target and self.user_target in self.df.columns:
                    # If they explicitly passed it in the initial screen, we can still override, but now we have dynamic prompt
                    target = self.user_target
            else:
                target = None
                await self._log("[dim]Path is unsupervised/explore. No target required.[/dim]")

            model_recs = orch_result.get("model_recommendations", ["RandomForest"])

            self._update_tree_node(tree, "Orchestrator", "done", [f"Path: {path}"])
            trace.finish(f"path={path}, target={target}")
            await self._update_reasoning("Orchestrator", orch_result.get("reasoning", "Autonomous routing decided."))
            await self._log("")

            # ── STEP 2: Domain Audit (Parallel) ──
            await self._log("[dim]─── Domain Audit (Parallel) ──────────────────────[/dim]")
            self._update_tree_node(tree, "Domain Sanity (Parallel)", "running")
            
            sanity_agent = SanityAgent(llm)
            leakage_agent = LeakageAgent(llm)
            
            # Profile the dataset first so SanityAgent has the numerical summary
            from beyondml.engine.profiler import DatasetProfiler
            profiler = DatasetProfiler(self.df, target_column=target)
            profile = profiler.run()

            # Run Sanity and Leakage in parallel
            sanity_task = sanity_agent.run(df_summary, profile.get("numerical_summary", {}), self._log)
            leakage_task = leakage_agent.run(target, self.description or "No description", {}, self._log)
            
            sanity_result, leakage_result = await asyncio.gather(sanity_task, leakage_task)
            
            # Rectify logical impossibilities
            for issue in sanity_result.get("issues", []):
                col = issue["column"]
                if col in self.df.columns:
                    vals = issue["invalid_values"]
                    self.df[col] = self.df[col].replace(vals, np.nan)
                    await self._log(f"  [yellow]Rectified[/yellow] {col}: converted {vals} to NaN")

            # Drop leakage columns
            leaked_applied = []
            for rec in leakage_result.get("recommendations", []):
                if rec["action"] == "drop" and rec["column"] in self.df.columns:
                    self.df = self.df.drop(columns=[rec["column"]])
                    leaked_applied.append(rec["column"])
                    await self._log(f"  [red]Dropped[/red] {rec['column']} (Leakage risk)")

            self._update_tree_node(tree, "Domain Sanity (Parallel)", "done", 
                [f"{len(sanity_result.get('issues', []))} sanity fixes", f"{len(leaked_applied)} leaks dropped"])
            await self._log("")

            # ── Data Drift Analysis ──
            if self.inference_path and os.path.exists(self.inference_path):
                await self._log("[dim]─── Data Drift Analysis ────────────────────────[/dim]")
                self._update_tree_node(tree, "Data Drift Analysis", "running")
                try:
                    ext = os.path.splitext(self.inference_path)[1].lower()
                    if ext in ['.xlsx', '.xls']:
                        df_inf = await asyncio.to_thread(pd.read_excel, self.inference_path)
                    elif ext == '.json':
                        df_inf = await asyncio.to_thread(pd.read_json, self.inference_path)
                    else:
                        df_inf = await asyncio.to_thread(pd.read_csv, self.inference_path)
                    
                    from beyondml.agents.drift_agent import DriftAgent
                    drift_agent = DriftAgent(llm)
                    drift_result = await drift_agent.run(self.df, df_inf, self._log)
                    self._update_tree_node(tree, "Data Drift Analysis", "done")
                    
                    if drift_result.get("status") == "success" and drift_result.get("drift_narrative"):
                        await self._update_reasoning("Data Drift Agent", drift_result["drift_narrative"])
                        
                except Exception as e:
                    await self._log(f"  [yellow]⚠ Drift Analysis skipped/failed: {e}[/yellow]")
                    self._update_tree_node(tree, "Data Drift Analysis", "done", ["Skipped"])
                await self._log("")
            else:
                self._update_tree_node(tree, "Data Drift Analysis", "done", ["Skipped (no inf. dataset)"])

            # ── STEP 3: Imputation ──
            await self._log("[dim]─── Imputation Agent ─────────────────────────────[/dim]")
            self._update_tree_node(tree, "Imputation", "running")
            
            # Re-profile to get current missing state
            profiler = DatasetProfiler(self.df, target_column=target)
            profile = profiler.run()
            
            impute_agent = ImputationAgent(llm)
            impute_result = await impute_agent.run(df_summary, profile.get("missing_analysis", {}), self._log)
            
            # Apply imputation
            for strat in impute_result.get("strategies", []):
                col = strat["column"]
                if col in self.df.columns:
                    mode = strat["strategy"]
                    if mode == "mean":
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
                    elif mode == "median":
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    elif mode == "mode":
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan)
                    elif mode == "constant":
                        self.df[col] = self.df[col].fillna(strat.get("fill_value", 0))
                    elif mode == "drop":
                        self.df = self.df.drop(columns=[col])

            self._update_tree_node(tree, "Imputation", "done")
            await self._log("")

            # ── STEP 4: Analysis (Parallel) ──
            await self._log("[dim]─── Analysis Layer (Parallel) ────────────────────[/dim]")
            self._update_tree_node(tree, "EDA Agent (Parallel)", "running")
            self._update_tree_node(tree, "Outlier Handler (Parallel)", "running")
            
            # Re-profile after imputation
            profiler = DatasetProfiler(self.df, target_column=target)
            profile = profiler.run()
            
            eda_agent = EDAAgent(llm)
            outlier_agent = OutlierAgent(llm)
            
            # Run EDA and Outlier detection in parallel
            # Note: OutlierAgent needs self.df, we'll pass it a copy or let it run on the shared df 
            # as EDA doesn't modify it.
            async def run_eda_and_capture():
                res = await eda_agent.run(self.df, profile, {"suggested_target": target}, self.description, self._log)
                self.interactive_charts = res.get("interactive_charts", [])
                return res

            eda_task = run_eda_and_capture()
            outlier_task = outlier_agent.run(self.df, profile.get("outlier_summary", {}), profile, self._log, self._get_user_input)
            
            eda_result, outlier_result = await asyncio.gather(eda_task, outlier_task)
            self.df = outlier_result["df"] # Update df with outlier changes

            # Process EDA Results (charts)
            for chart_name, chart_str in eda_result.get("rendered_charts", []):
                await self._log(f"\n[bold magenta]── {chart_name} ──[/bold magenta]")
                try:
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
            
            self._update_tree_node(tree, "EDA Agent (Parallel)", "done",
                [f"Target: {confirmed_target}", f"{len(eda_result.get('eda_insights', []))} insights"])
            self._update_tree_node(tree, "Outlier Handler (Parallel)", "done",
                [f"Strategy: {outlier_result['outlier_strategy']}"])
            await self._log("")

            if path in ["explore", "dimensionality_reduction"]:
                self._update_tree_node(tree, "Explore / PCA", "done")
                await self._log("\n[bold green]✓ Exploration and Dimensionality Reduction complete![/bold green]")
                return

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

                if path == "supervised":
                    # ── STEP 5: GA Trainer (supervised) ──
                    await self._log("[dim]─── GA Trainer ──────────────────────── [running] ──[/dim]")
                    self._update_tree_node(tree, "GA Trainer (Supervised)", "running")

                    # Re-profile with new features
                    profiler = DatasetProfiler(self.df, target_column=confirmed_target)
                    try:
                        profile = profiler.run()
                        if not profile:
                            profile = {"target_analysis": {"target_type": "classification"}, "numerical_summary": {}}
                    except Exception as e:
                        await self._log(f"[yellow]Warning profiling failed: {e}. Using fallback profile.[/yellow]")
                        profile = {"target_analysis": {"target_type": "classification"}, "numerical_summary": {}}
                    
                    if "target_analysis" not in profile or profile["target_analysis"] is None:
                        profile["target_analysis"] = {"target_type": "classification"}

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
                    best_params = ga_result["best_params"]
                    model_type = ga_result["model_type"]

                    self._update_tree_node(tree, "GA Trainer (Supervised)", "done",
                        [f"Best: {ga_result['best_cv_score']:.4f}", f"Model: {ga_result['model_type']}"])

                    # ── STEP 5.5: Ensemble Agent ──
                    top_genomes = ga_result.get("top_genomes", [])
                    prebuilt_model = None
                    if self.use_ensemble and len(top_genomes) > 1:
                        await self._log("\n[dim]─── Ensemble Agent ────────────────────────────[/dim]")
                        self._update_tree_node(tree, "Ensemble", "running")
                        
                        from beyondml.agents.ensemble_agent import EnsembleAgent
                        ensemble_agent = EnsembleAgent(llm)
                        ensemble_result = await ensemble_agent.run(
                            df=self.df,
                            target_column=confirmed_target,
                            profile=profile,
                            top_genomes=top_genomes,
                            problem_type=profile["target_analysis"]["target_type"],
                            log=self._log,
                            strategy="stacking"
                        )
                        
                        ens_score = ensemble_result["test_score"]
                        self._update_tree_node(tree, "Ensemble", "done", [f"Score: {ens_score:.4f}"])
                        
                        if ens_score > ga_result["best_cv_score"]:
                            await self._log(f"  [bold green]Ensemble outperforms single best model![/bold green]")
                            best_params = {"strategy": "stacking", "base_models": ensemble_result["base_models"]}
                            model_type = ensemble_result["model_type"]
                            from beyondml.engine.ensemble import EnsembleEngine
                            engine = EnsembleEngine(profile["target_analysis"]["target_type"])
                            prebuilt_model = engine.build_stacking(top_genomes)
                
                elif path == "deep_learning":
                    # ── STEP 5: DL Trainer ──
                    await self._log("[dim]─── DL Trainer ──────────────────────── [running] ──[/dim]")
                    self._update_tree_node(tree, "DL Trainer (Neural Net)", "running")

                    dl_agent = DeepLearningAgent(llm)
                    dl_result = await dl_agent.run(
                        df=self.df,
                        target_column=confirmed_target,
                        problem_type=profile["target_analysis"]["target_type"],
                        log=self._log,
                        epochs=10
                    )
                    
                    # Mock GA result structure for Evaluator
                    ga_result = {
                        "best_params": {},
                        "model_type": "SimpleMLP",
                        "best_cv_score": dl_result["test_score"]
                    }
                    best_params = {}
                    model_type = "SimpleMLP"

                    self._update_tree_node(tree, "DL Trainer (Neural Net)", "done",
                        [f"Acc: {dl_result['test_score']:.4f}"])
                    prebuilt_model = None

                await self._log("")

                # ── STEP 6: Evaluator ──
                await self._log("[dim]─── Evaluator ───────────────────────────────────[/dim]")
                self._update_tree_node(tree, "Evaluator", "running")

                eval_agent = EvaluatorAgent(llm)
                eval_result = await eval_agent.run(
                    df=self.df,
                    target_column=confirmed_target,
                    profile=profile,
                    best_params=best_params,
                    model_type=model_type,
                    problem_type=profile["target_analysis"]["target_type"],
                    log=self._log,
                    test_size=self.test_size,
                    prebuilt_model=locals().get("prebuilt_model", None)
                )

                self._update_tree_node(tree, "Evaluator", "done",
                    [f"Score: {eval_result['test_score']:.4f}"])
                await self._update_reasoning("Evaluator", eval_result.get("eval_narration", "Final model performance validated."))
                
                # ── STEP 6.5: Explainability ──
                import joblib
                from beyondml.agents.explainability_agent import ExplainabilityAgent
                
                await self._log("\n[dim]─── Explainability Agent ────────────────────────[/dim]")
                self._update_tree_node(tree, "Explainability", "running")
                
                try:
                    fitted_pipe = joblib.load(eval_result["model_path"])
                    X_eval = self.df.drop(columns=[confirmed_target]) if confirmed_target in self.df.columns else self.df
                    
                    explain_agent = ExplainabilityAgent(llm)
                    explain_result = await explain_agent.run(
                        model_pipeline=fitted_pipe,
                        X_eval=X_eval,
                        target_column=confirmed_target,
                        problem_type=profile["target_analysis"]["target_type"],
                        log=self._log
                    )
                    eval_result["xai_result"] = explain_result
                    self._update_tree_node(tree, "Explainability", "done")
                except Exception as e:
                    await self._log(f"  [yellow]⚠ Explainability skipped: {e}[/yellow]")
                    self._update_tree_node(tree, "Explainability", "done", ["Skipped"])
                
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
                    
                    # Store on self for the export function
                    self.best_eval_result = best_eval_result
                    self.best_params = best_params
                    self.best_model_type = model_type
                    self.confirmed_target = confirmed_target
                    self.problem_type = profile["target_analysis"]["target_type"]
                    
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

    def action_view_charts(self):
        if not hasattr(self, 'interactive_charts') or not self.interactive_charts:
            self.notify("No interactive charts have been generated yet.", severity="warning")
            return
            
        import webbrowser
        self.notify(f"Opening {len(self.interactive_charts)} charts in browser...", severity="information")
        for chart_path in self.interactive_charts:
            try:
                webbrowser.open('file://' + chart_path)
            except Exception as e:
                self.notify(f"Failed to open chart: {e}", severity="error")

    def action_save(self):
        self.notify("State saved!", severity="information")

    @work(thread=False)
    async def action_export(self):
        if not hasattr(self, 'best_eval_result') or not self.best_eval_result:
            self.notify("No model has been fully trained yet to export.", severity="warning")
            return
            
        self.notify("Generating Jupyter Notebook...", severity="information")
        try:
            from beyondml.agents.codegen_agent import CodeGenAgent
            from beyondml.llm import get_llm_provider
            
            # Use the same LLM
            if self.llm_choice == "groq":
                from beyondml.llm.groq_provider import GroqProvider
                llm = GroqProvider()
            else:
                from beyondml.llm.ollama_provider import OllamaProvider
                llm = OllamaProvider()
                
            codegen = CodeGenAgent(llm)
            out_path = await codegen.run(
                dataset_path=self.dataset_path,
                target_column=self.confirmed_target,
                problem_type=self.problem_type,
                best_params=self.best_params,
                model_type=self.best_model_type,
                eval_result=self.best_eval_result,
                log=self._log,
                test_size=self.test_size
            )
            self.notify(f"Exported to {out_path}!", severity="success")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
            await self._log(f"  [red]⚠ Export error: {e}[/red]")

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
        if not check_config():
            self.push_screen(ConfigScreen(), self._after_config)
        else:
            self.push_screen(WelcomeScreen())

    def _after_config(self, success: bool):
        if success:
            self.push_screen(WelcomeScreen())
        else:
            # If they canceled setup somehow, we can't really proceed
            self.exit()


if __name__ == "__main__":
    app = BeyondMLApp()
    app.run()
