"""
Giga AI – Three-layer hierarchical bot system.

Layers
------
1. Main Bot (not included – user-wired) – top-level goal orchestrator
2. Manager Bots                         – task orchestrators, spawn sub-bots
3. Sub Bots                             – atomic workers (scraper, selenium, …)

Brain modules drive the Main Bot:
  PerceptionBrain  – goal intake & health monitoring
  PlanningBrain    – LLM-powered goal decomposition
  ExecutionBrain   – manager lifecycle management
  LearningBrain    – global memory & failure analysis
"""

__version__ = "0.1.0"
