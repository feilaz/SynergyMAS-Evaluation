# SynergyMAS: Enhancing LLM Capabilities for Complex Tasks

## Introduction

SynergyMAS is a Multi-Agent System (MAS) framework designed to enhance Large Language Models (LLMs) for complex tasks, integrating logical reasoning, Retrieval-Augmented Generation (RAG), and Theory of Mind (ToM). 
This repository contains a modified version for comparative evaluation. To enable fair comparisons with methods handling shorter interactions, conversation stage tracking functionality have been removed. 
For instance, in applying Lean Startup methodology, this version evaluates only the "Learn" step instead of all five steps (see example at the end of README).

## Project Structure

- `main.py`: Entry point for the SynergyMAS framework
- `agents.py`: Definitions and implementations of specialized agents
- `rag.py`: Retrieval-Augmented Generation system
- `tools.py`: Knowledge base system and logical reasoning tools
- `config_loader.py`: Configuration management
- `prompts.py`: System prompts for agents

## Requirements

- Python 3.8+
- Neo4j Database
- OpenAI API key
- LangChain API key (optional, for tracing)

## Installation

1. Clone the repository

2. Install dependencies:
pip install -r requirements.txt

3. Set up your Neo4j database and note down the connection details.
This can be done at: https://sandbox.neo4j.com/

## Configuration

1. Copy the example configuration file:

2. Edit `config.ini` with your specific settings:
- Add your OpenAI API key
- Set your Neo4j database connection details
- Configure paths for RAG data and Chroma DB
- Set LangChain tracing options (if used)

## Preparing RAG Data

Place your profession-specific data in the following directories:
- `rag/MRA/`: Market Research Analyst data
- `rag/PD/`: Product Designer data
- `rag/SM/`: Sales Manager data

## Running the Program

Execute the main script with your problem statement:

python main.py "Your problem statement here"

For example:
python main.py """Analyze the Smart Home Energy Management System (SHEMS) MVP 6-week trial data:

12.3% average energy savings (high variability)
78% weekly app usage, 45% AI feature usage
Top features: Real-time monitoring, remote control
NPS: 32
3 minor outages, 15% device integration issues
65% support tickets on device integration
20% higher engagement in moderate climates
Competitor launched product with gamification
Provide:

Top 3 insights from the data
Analysis of energy savings variability
Assessment of AI feature performance with improvement suggestions
Evaluation of competitive position
Top 3 priorities for next iteration
Justify your answers with data-driven reasoning."""