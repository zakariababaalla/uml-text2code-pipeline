# uml-text2code-pipeline
Hybrid NLP–DSL–MDA pipeline for the automatic generation of UML diagrams and Python code from textual specifications.
# UML-Text2Code Pipeline  
A hybrid NLP–DSL–MDA pipeline for automatic generation of UML class diagrams and Python code from natural language specifications.  

---

## Overview
This project implements a model-driven engineering (MDA) pipeline that transforms natural language requirements into executable software artifacts.  
The pipeline is structured in four main stages:  

1. **Textual Specification (CIM)**  
   Natural language requirements (functional requirements, use cases, business rules).  

2. **Semantic Extraction (NLP)**  
   A Transformer-based NER model extracts UML entities (classes, attributes, methods, relations).  

3. **Intermediate Structuring (DSL / PIM)**  
   Extracted entities are organized in a domain-specific language (DSL) that ensures transparency, validation, and manual corrections.  

4. **Transformation & Code Generation (PSM)**  
   The DSL is automatically transformed into UML class diagrams and then into Python code.  

---

## 📂 Project Structure
uml-text2code-pipeline/
│── dataset.txt # Training corpus
│── train_bert.py # Script to fine-tune BERT for UML entity extraction
│── uml_extract.py # Extract UML entities from text using trained model
│── uml_to_code.py # Generate Python code from UML representation
│── uml_gui.py # Graphical interface (uses logo.jpg)
│── logo.jpg # Logo used in GUI
│── README.md # Project documentation                
