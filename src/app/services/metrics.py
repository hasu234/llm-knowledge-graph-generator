"""
Knowledge graph metrics calculation (ICR and IPR)
"""

from typing import List, Tuple, Set

def extract_unique_classes_from_triples(triples_list: List[Tuple[str, str, str]]) -> Set[str]:
    """
    Extract unique classes from the generated triples based on subjects and objects
    
    Args:
        triples_list: List of (subject, predicate, object) tuples
        
    Returns:
        Set of unique class URIs
    """
    unique_classes = set()
    
    for subject, predicate, obj in triples_list:
        # Add subjects that are DBpedia ontology classes/resources
        if subject.startswith("http://dbpedia.org/"):
            unique_classes.add(subject)
        
        # Add objects that are DBpedia ontology classes/resources (not literals)
        # Exclude literal values (those with quotes, datatype markers, or language tags)
        if (obj.startswith("http://dbpedia.org/") and 
            not obj.startswith('"') and 
            '^^' not in obj and 
            '@' not in obj):
            unique_classes.add(obj)
    
    return unique_classes

def find_instantiated_classes(triples_list: List[Tuple[str, str, str]], unique_classes: Set[str]) -> Set[str]:
    """
    Find classes that have instances (appear as objects in rdf:type relations)
    
    Args:
        triples_list: List of (subject, predicate, object) tuples
        unique_classes: Set of unique classes to check against
        
    Returns:
        Set of instantiated class URIs
    """
    instantiated_classes = set()
    
    for subject, predicate, obj in triples_list:
        # Look for rdf:type predicates or predicates containing 'type'
        if 'type' in predicate.lower() or 'rdf:type' in predicate:
            # Object is the class being instantiated
            if obj in unique_classes:
                instantiated_classes.add(obj)
        
        # Also look for RDFS predicates that indicate class relationships
        if 'rdfs:' in predicate.lower() or 'rdfs' in predicate.lower():
            if obj in unique_classes:
                instantiated_classes.add(obj)
    
    return instantiated_classes

def calculate_icr_metric(triples_list: List[Tuple[str, str, str]]) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate ICR (Instantiated Class Ratio) metric
    ICR = n(IC) / n(C)
    where:
    - n(C) = total number of unique classes in the ontology/triples
    - n(IC) = number of classes that have instances
    
    Args:
        triples_list: List of (subject, predicate, object) tuples
        
    Returns:
        Tuple of (ICR score, unique classes, instantiated classes)
    """
    # Step 1: Extract all unique classes from triples
    unique_classes = extract_unique_classes_from_triples(triples_list)
    
    # Step 2: Find classes that have instances
    instantiated_classes = find_instantiated_classes(triples_list, unique_classes)
    
    # Step 3: Calculate ICR
    n_C = len(unique_classes)  # Total number of classes
    n_IC = len(instantiated_classes)  # Number of instantiated classes
    
    if n_C == 0:
        icr = 0.0
    else:
        icr = n_IC / n_C
    
    return icr, unique_classes, instantiated_classes

def extract_unique_properties_from_triples(triples_list: List[Tuple[str, str, str]]) -> Set[str]:
    """
    Extract unique properties (predicates) from the generated triples
    
    Args:
        triples_list: List of (subject, predicate, object) tuples
        
    Returns:
        Set of unique property URIs
    """
    unique_properties = set()
    
    for subject, predicate, obj in triples_list:
        # Add the predicate to unique properties
        unique_properties.add(predicate)
    
    return unique_properties

def calculate_ipr_metric(triples_list: List[Tuple[str, str, str]]) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate IPR (Instantiated Property Ratio) metric
    IPR = n(IP) / n(P)
    where:
    - n(P) = total number of unique properties found in the triples
    - n(IP) = number of properties that are from RDF/RDFS/OWL standard namespaces
    
    This measures how many standard semantic web properties are used
    compared to all properties in the knowledge graph.
    
    Args:
        triples_list: List of (subject, predicate, object) tuples
        
    Returns:
        Tuple of (IPR score, unique properties, standard namespace properties)
    """
    # Step 1: Extract all unique properties from triples
    unique_properties = extract_unique_properties_from_triples(triples_list)
    
    # Step 2: Count instantiated properties (properties from standard semantic web namespaces)
    instantiated_properties = set()
    
    # Standard semantic web namespaces
    standard_namespaces = [
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#',  # RDF
        'http://www.w3.org/2000/01/rdf-schema#',        # RDFS
        'http://www.w3.org/2002/07/owl#',               # OWL
        'http://xmlns.com/foaf/0.1/',                   # FOAF
        'http://purl.org/dc/elements/1.1/',             # Dublin Core
        'http://purl.org/dc/terms/',                    # Dublin Core Terms
    ]
    
    # Short form prefixes for standard namespaces
    standard_prefixes = ['rdf:', 'rdfs:', 'owl:', 'foaf:', 'dc:', 'dct:']
    
    for _, predicate, _ in triples_list:
        # Check for full URI form
        for namespace in standard_namespaces:
            if namespace in predicate:
                instantiated_properties.add(predicate)
                break
        else:
            # Check for short form (rdf:type, foaf:name, etc.)
            for prefix in standard_prefixes:
                if predicate.startswith(prefix):
                    instantiated_properties.add(predicate)
                    break
    
    # Step 3: Calculate IPR
    n_P = len(unique_properties)  # Total number of unique properties
    n_IP = len(instantiated_properties)  # Number of standard namespace properties
    
    if n_P == 0:
        ipr = 0.0
    else:
        ipr = n_IP / n_P
    
    return ipr, unique_properties, instantiated_properties
