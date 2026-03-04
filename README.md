# 🚀 BeyondML

**AI Agent Orchestration Platform — Terminal-native AutoML with LLM-powered agents and Genetic Algorithm optimization.**

BeyondML is an autonomous machine learning pipeline that uses multiple specialized AI agents (EDA, Feature Engineering, GA Optimization, Reflection) orchestrated through an LLM to profile, preprocess, train, and evaluate models — all from a beautiful terminal UI.

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/Riteesh-2k6/beyondml.git
cd beyondml

# Install (editable mode with dev tools)
pip install -e ".[dev]"

# Launch the TUI
beyondml run
```

---

## 🏗️ Architecture

```
tui_app.py                   ← Textual TUI (3-column pipeline screen)
beyondml/
├── cli.py                   ← CLI entry point (beyondml run / benchmark)
├── state.py                 ← MLState — shared pipeline state
├── charts.py                ← Plotext chart rendering
├── llm/
│   ├── base.py              ← Abstract LLMProvider
│   ├── groq_provider.py     ← Groq cloud LLM
│   └── ollama_provider.py   ← Ollama local LLM
├── engine/
│   ├── profiler.py          ← DatasetProfiler + TargetIdentifier + ORI
│   ├── supervised.py        ← SupervisedPipeline (baselines + final)
│   ├── unsupervised.py      ← UnsupervisedPipeline (KMeans, DBSCAN, PCA)
│   ├── genetic.py           ← Genome + GeneticModelOptimizer
│   ├── metrics.py           ← Classification & regression metrics
│   └── benchmarker.py       ← PMLB benchmark runner
└── agents/
    ├── orchestrator.py      ← LLM-powered path router
    ├── eda_agent.py         ← Exploratory Data Analysis
    ├── outlier_agent.py     ← Outlier detection & handling
    ├── feature_agent.py     ← LLM feature engineering
    ├── ga_trainer.py        ← GA evolution + TUI progress
    ├── evaluator_agent.py   ← Final model evaluation + narration
    └── reflection_agent.py  ← Iterative improvement loop
```

### Agent Pipeline Flow

```
Dataset → Orchestrator → EDA Agent → Outlier Agent → Feature Agent
       → GA Trainer → Evaluator → Reflection Agent → (loop or finish)
```

---

## 🛠️ Usage

### TUI (Interactive)

```bash
beyondml run
```

Launches the full interactive pipeline with dataset selection, agent orchestration, and real-time progress.

### Benchmark (PMLB)

```bash
# Default datasets (titanic, car)
beyondml benchmark

# Custom datasets
beyondml benchmark -d adult -d iris
```

### Environment Configuration

Create a `.env` file:

```env
# LLM Provider: "ollama" (default) or "groq"
LLM_PROVIDER=ollama

# Required only if using Groq
GROQ_API_KEY=your-api-key-here
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=beyondml --cov-report=term-missing
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | ML models & preprocessing |
| `textual` | Terminal UI framework |
| `plotext` | Terminal-native charts |
| `groq` / `requests` | LLM providers (Groq / Ollama) |
| `pmlb` | Penn ML Benchmark datasets |
| `click` | CLI framework |

---

## 📄 License

MIT
