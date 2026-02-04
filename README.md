# Quant Trading Simulation Framework

A modular, research-oriented trading simulation framework designed to demonstrate
systematic strategy development, execution, and performance evaluation.

This project emphasizes clean architecture, reproducibility, and risk-aware
portfolio management — aligned with professional quant research workflows.

---

## Project Overview

This system simulates a **systematic mean-reversion strategy** using rolling
z-scores computed on historical price data. The framework is structured to
separate concerns clearly:

- Data ingestion
- Signal generation
- Portfolio execution
- Risk management
- Performance evaluation

The goal is not alpha claims, but **demonstrating sound quantitative engineering
principles**.

---

## Strategy Logic

**Signal**
- Rolling mean and standard deviation over a configurable lookback window
- Z-score = (price − rolling mean) / rolling std
- BUY when z-score < lower threshold
- SELL when z-score > upper threshold

**Execution & Risk**
- Position sizing capped by max portfolio allocation
- Transaction cost modeling (bps)
- No leverage
- Long-only baseline

**Metrics**
- Annualized Sharpe Ratio
- Maximum Drawdown

---

## Repository Structure


