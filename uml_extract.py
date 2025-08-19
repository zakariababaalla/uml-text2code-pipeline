#!/usr/bin/env python3

import re
import os
import logging
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig
from safetensors.torch import load_file
import torch

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Preprocess text
def preprocess_text(text: str) -> str:
    """Preprocess and clean the input text."""
    logger.debug("Preprocessing input text.")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    logger.debug(f"Cleaned text: {text[:100]}...")
    return text

# Normalize class names
def normalize_class_name(class_name: str) -> str:
    """Normalize class names to a consistent format."""
    normalized = class_name.lower()
    if normalized.endswith('s') and len(normalized) > 1:
        normalized = normalized[:-1]
    return normalized.capitalize()

# Load model and tokenizer
MODEL_PATH = os.path.abspath('G:/bert/my-finetuned-model')
necessary_files = ["config.json", "tokenizer.json", "vocab.txt", "model.safetensors"]
for file in necessary_files:
    if not os.path.exists(os.path.join(MODEL_PATH, file)):
        raise FileNotFoundError(f"Missing file: {file}")

config = BertConfig.from_pretrained(MODEL_PATH)
model = BertForTokenClassification(config)
model_weights = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
model.load_state_dict(model_weights)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

logger.info("Model and tokenizer loaded successfully.")

# Label mapping
id2label = {
    0: 'O', 1: 'B_CLASS_SOURCE', 2: 'I_CLASS_SOURCE', 3: 'B_CLASS_TARGET', 4: 'I_CLASS_TARGET',
    5: 'B_ATTRIBUTE', 6: 'I_ATTRIBUTE', 7: 'B_METHOD', 8: 'I_METHOD',
    9: 'B_ASSOCIATION', 10: 'I_ASSOCIATION', 11: 'B_AGGREGATION', 12: 'I_AGGREGATION',
    13: 'B_COMPOSITION', 14: 'I_COMPOSITION', 15: 'B_INHERITANCE', 16: 'I_INHERITANCE'
}

# Merge subtokens into whole words with labels
def merge_subtokens(tokens, labels):
    """Merge subtokens into complete tokens and associate labels."""
    merged_tokens, merged_labels = [], []
    buffer, buffer_label = "", None

    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            buffer += token[2:]
        else:
            if buffer:
                merged_tokens.append(buffer)
                merged_labels.append(buffer_label)
            buffer, buffer_label = token, label

    if buffer:
        merged_tokens.append(buffer)
        merged_labels.append(buffer_label)

    return merged_tokens, merged_labels

RELATION_PRIORITY = {"composition": 3, "aggregation": 2, "inheritance": 1, "association": 0}

def filter_strongest_relations(relations):
    """Keep only the strongest relation between two classes."""
    strongest_relations = {}
    for r in relations:
        key = (r['source'], r['target'])
        reverse_key = (r['target'], r['source'])
        cur_priority = RELATION_PRIORITY.get(r['type'], -1)

        if key in strongest_relations:
            if cur_priority > RELATION_PRIORITY.get(strongest_relations[key]['type'], -1):
                strongest_relations[key] = r
        elif reverse_key in strongest_relations:
            if cur_priority > RELATION_PRIORITY.get(strongest_relations[reverse_key]['type'], -1):
                strongest_relations[reverse_key] = r
        else:
            strongest_relations[key] = r

    return list(strongest_relations.values())

def extract_uml_elements(text: str) -> dict:
    """Extract UML classes, attributes, methods, and relationships, chunked for long sequences."""
    words = text.split()
    max_len = 510  # BERT supports max 512 with [CLS] and [SEP]

    all_tokens = []
    all_labels = []

    for i in range(0, len(words), max_len):
        chunk = words[i:i + max_len]
        encoding = tokenizer(chunk, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**encoding)

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        predicted_labels = [id2label[p.item()] for p in predictions]
        token_list = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        token_list = [t.replace("Ġ", "") for t in token_list]
        merged_tokens, merged_labels = merge_subtokens(token_list, predicted_labels)

        if len(merged_tokens) != len(merged_labels):
            logger.warning("Skipping mismatched token/label sequence.")
            continue

        all_tokens.extend(merged_tokens)
        all_labels.extend(merged_labels)

    if not all_tokens:
        return {'classes': {}, 'relations': []}

    uml_classes = {}
    seen_classes = set()
    relations = []
    current_class = None
    current_attr = None
    current_method = None
    last_class = None
    current_relation = None
    relation_detected = False

    for token, label in zip(all_tokens, all_labels):
        if label in ['B_CLASS_SOURCE', 'B_CLASS_TARGET']:
            normalized_class = normalize_class_name(token)
            if normalized_class not in seen_classes:
                uml_classes[normalized_class] = {'attributes': [], 'methods': []}
                seen_classes.add(normalized_class)
                logger.info(f"Class detected: {normalized_class}")

            current_class = normalized_class
            if relation_detected and last_class:
                relations.append({
                    'source': last_class,
                    'target': current_class,
                    'type': current_relation
                })
                logger.info(f"Relation added: {last_class} -> {current_class} (Type: {current_relation})")
                relation_detected = False
                current_relation = None

            last_class = current_class

        elif label == 'B_ATTRIBUTE' and current_class:
            current_attr = token
            uml_classes[current_class]['attributes'].append(current_attr)
            logger.info(f"Attribute detected for class {current_class}: {current_attr}")

        elif label == 'I_ATTRIBUTE' and current_attr:
            if uml_classes[current_class]['attributes']:
                uml_classes[current_class]['attributes'][-1] += " " + token

        elif label == 'B_METHOD' and current_class:
            current_method = token
            uml_classes[current_class]['methods'].append(current_method)
            logger.info(f"Method detected for class {current_class}: {current_method}")

        elif label == 'I_METHOD' and current_method:
            if uml_classes[current_class]['methods']:
                uml_classes[current_class]['methods'][-1] += " " + token

        elif label in ['B_ASSOCIATION', 'B_AGGREGATION', 'B_COMPOSITION', 'B_INHERITANCE']:
            current_relation = label.split("_")[-1].lower()
            relation_detected = True

    relations = filter_strongest_relations(relations)

    logger.info(f"Classes detected: {uml_classes}")
    logger.info(f"Relations detected: {relations}")

    return {'classes': uml_classes, 'relations': relations}

