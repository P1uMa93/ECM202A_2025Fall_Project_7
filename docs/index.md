---
layout: default
title: "Structured Reasoning for Sensor Data: Human Activity Recognition with GraphRAG"
---

# **Structured Reasoning for Sensor Data: Human Activity Recognition with GraphRAG**

![Project Banner](./assets/img/banner-placeholder.png)  
<sub>*(Optional: Replace with a conceptual figure or meaningful image.)*</sub>

---

## 👥 **Contributors**

- Richard Xiao (UCLA MEng Student) 
- Mentor - Liying Han (UCLA ECE PhD)

---

## 📝 **Abstract**

Human Activity Recognition (HAR) from sensor data is critical for healthcare monitoring, smart homes, and mobile interaction, yet traditional approaches using Large Language Models lack transparency and provide black-box predictions that impede systematic debugging and improvement. This project investigates whether Graph-Augmented Retrieval-Augmented Generation (GraphRAG) can improve HAR accuracy and interpretability by constructing knowledge graphs that explicitly model entities, relationships, and hierarchical community structures from sensor data. We implemented a GraphRAG-based HAR system using the UCI Human Activity Recognition Dataset with nano-graphrag and Ollama models (gpt-oss:20b and 120b), comparing performance between global and local query modes. Our experiments revealed that local search mode significantly outperforms global search, with global mode failing to provide answers approximately 15% of the time due to insufficient context aggregation, while local mode demonstrated robust performance through entity-centric graph traversal. These findings demonstrate that GraphRAG's structured reasoning approach provides engineers with interpretable, traceable decision paths that transform HAR from opaque classification to auditable systems where failure modes can be systematically identified and addressed.

---

## 📑 **Submissions**

- [Midterm Checkpoint Slides](http://)  
- [Final Presentation Slides](http://)
- [Final Report](report)

---

