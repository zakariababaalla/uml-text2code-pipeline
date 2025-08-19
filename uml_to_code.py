#!/usr/bin/env python3

def generate_dsl(uml_data: dict) -> str:
    """Generate DSL from UML elements."""
    dsl = ""
    for class_name, details in uml_data['classes'].items():
        dsl += f"Class {class_name}:\n"
        if details['attributes']:
            dsl += "  Attributes: " + ", ".join(details['attributes']) + "\n"
        if details['methods']:
            dsl += "  Methods: " + ", ".join(details['methods']) + "\n"
    for rel in uml_data['relations']:
        dsl += f"Relation: {rel['source']} {rel['type']} {rel['target']}\n"
    return dsl


def parse_dsl_to_uml(dsl: str) -> dict:
    """Parse a DSL block into UML class structure and relations,
       with correction rules and anomaly reporting."""

    classes = {}
    relations = []
    current_class = None
    validation_report = []

    for line_no, line in enumerate(dsl.strip().splitlines(), start=1):
        original_line = line
        line = line.strip()

        # Basic automatic correction: normalize case and keywords
        line = line.replace("attributes", "Attributes").replace("methods", "Methods")
        line = line.replace("relation", "Relation").replace("class", "Class")

        if not line:
            continue

        # Detect a new class
        if line.startswith("Class "):
            class_name = line[len("Class "):].strip(":")
            if not class_name:
                validation_report.append(
                    f"[Line {line_no}] Class name missing in '{original_line}'"
                )
                continue
            current_class = class_name
            classes[current_class] = {"attributes": [], "methods": []}

        # Add attributes
        elif line.startswith("Attributes:"):
            attrs = line[len("Attributes:"):].strip()
            if current_class:
                if attrs:
                    attr_list = [a.strip() for a in attrs.split(",") if a.strip()]
                    classes[current_class]["attributes"].extend(attr_list)
                else:
                    validation_report.append(
                        f"[Line {line_no}] Empty attribute list in '{current_class}'"
                    )

        # Add methods 
        elif line.startswith("Methods:"):
            methods = line[len("Methods:"):].strip()
            if current_class:
                if methods:
                    method_list = [m.strip() for m in methods.split(",") if m.strip()]
                    classes[current_class]["methods"].extend(method_list)
                else:
                    validation_report.append(
                        f"[Line {line_no}] Empty method list in '{current_class}'"
                    )

        # Add relations (to be validated later) 
        elif line.startswith("Relation:"):
            parts = line[len("Relation:"):].strip().split()
            if len(parts) == 3:
                source, relation_type, target = parts
                relations.append({
                    "source": source,
                    "target": target,
                    "type": relation_type.lower()
                })
            else:
                validation_report.append(
                    f"[Line {line_no}] Invalid relation format in '{original_line}'"
                )
        else:
            validation_report.append(
                f"[Line {line_no}] Unrecognized line format: '{original_line}'"
            )

    # Remove invalid relations (missing class)
    valid_relations = []
    for r in relations:
        if r["source"] in classes and r["target"] in classes:
            valid_relations.append(r)
        else:
            validation_report.append(
                f"Relation dropped: '{r['source']} {r['type']} {r['target']}' "
                f"(missing class definition)"
            )

    return {
        "classes": classes,
        "relations": valid_relations,
        "validation_report": validation_report
    }


def generate_python_code_from_dsl(dsl: str) -> str:
    """Generate Python code from DSL format."""
    uml_data = parse_dsl_to_uml(dsl)
    return generate_python_code(uml_data)


def generate_python_code(uml_data: dict) -> str:
    """Generate Python source code from UML data."""
    code = ""
    for cls, details in uml_data['classes'].items():
        code += f"class {cls}:\n"
        if not details['attributes'] and not details['methods']:
            code += "    pass\n\n"
            continue

        # Constructor with attributes 
        code += "    def __init__(self"
        for attr in details['attributes']:
            code += f", {attr}: str = None"
        code += "):\n"
        for attr in details['attributes']:
            code += f"        self.{attr} = {attr}\n"
        if not details['attributes']:
            code += "        pass\n"
        code += "\n"

        # Methods
        for method in details['methods']:
            code += f"    def {method}(self):\n        pass\n\n"
    return code
