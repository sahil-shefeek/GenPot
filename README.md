# 🛡️ GenPot: An LLM-Powered, Multi-Protocol Web API & SMTP Honeypot

[![Python Tests](https://github.com/sahil-shefeek/GenPot/actions/workflows/python-tests.yml/badge.svg)](https://github.com/sahil-shefeek/GenPot/actions/workflows/python-tests.yml)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

GenPot is a next-generation, high-interaction honeypot implemented as an **LLM-Powered, Multi-Protocol Web API & SMTP Honeypot**. 

Traditional honeypots use pre-scripted, canned responses that sophisticated adversaries easily identify. GenPot solves this by implementing the **DecoyPot methodology**. It leverages a **Retrieval-Augmented Generation (RAG)** framework and **Large Language Models (LLMs)** to act as the honeypot's "brain." This allows GenPot to generate dynamic, context-aware, and highly realistic decoy responses on the fly, effectively engaging attackers while preventing LLM hallucinations by grounding responses in real API and protocol documentation.

---

## 🏗️ Architecture (The 4-Layer Model)

GenPot has recently undergone a massive architectural shift to support multiple protocols and enterprise-grade telemetry. The system is divided into four distinct layers:

### Layer 1: Ingress (Emulators)
The ingress layer handles incoming network traffic across different protocols and normalizes it into standard `UnifiedRequest` objects.
- **HTTP Emulator** (`http_emulator.py`): Built with FastAPI, it captures and normalizes RESTful API attacks.
- **SMTP Emulator** (`smtp_emulator.py`): Built with AsyncIO, it acts as a mail server honeypot, capturing SMTP commands and payloads.

### Layer 2: Core Engine
The centralized processing hub of the honeypot.
- **`GenPotEngine`**: The orchestrator that processes `UnifiedRequest` objects.
- **`PromptStrategy` Pattern**: Implements dynamic routing for LLM interactions. It routes requests to the appropriate strategy (e.g., `HttpPromptStrategy`, `SmtpPromptStrategy`) to generate protocol-specific, context-aware prompts.
- **`StateManager`**: Manages interaction context, including a transient `sessions` scope to maintain state for individual TCP connections during multi-step attacks.

### Layer 3: Telemetry
Instead of fragmented logs, GenPot standardizes all logging using the **Elastic Common Schema (ECS)**. All network events, protocol-specific metadata, and AI generation metrics are structured into ECS-compliant JSON lines and written to `logs/honeypot.jsonl`.

### Layer 4: Dashboard
The legacy Streamlit UI has been deprecated in favor of a robust **ELK Stack** (Elasticsearch, Logstash/Filebeat, Kibana). Filebeat ingests the ECS logs, and Kibana provides a powerful SOC dashboard to visualize attacker tactics, techniques, and procedures (TTPs) in real-time.

---

## 🚀 Quickstart Guide

Follow these steps to deploy the full GenPot system (Honeypot + SIEM) locally.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Start the SIEM (ELK Stack):**
   ```bash
   docker compose up -d
   ```

3. **Start the Honeypot:**
   ```bash
   uv run python -m server.main
   ```

4. **Access Kibana:**
   Open your browser and navigate to the SOC dashboard:
   [http://localhost:5601](http://localhost:5601)

---

## ⚔️ Testing & Attack Simulation

GenPot includes automated attack scripts to verify functionality and simulate adversary behavior. Run these in a separate terminal while the honeypot is active.

- **Simulate HTTP Attacks:**
  ```bash
  uv run python -m scripts.test_scripts.test_live_attacks
  ```

- **Simulate SMTP Attacks:**
  ```bash
  uv run python -m scripts.test_scripts.test_smtp_attack
  ```

---

## 📜 Citation

This project is an implementation based on the following research paper:
> Sezgin, A., & Boyacı, A. (2025). DecoyPot: A large language model-driven web API honeypot for realistic attacker engagement. *Computers & Security*, *154*, 104458.

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.
