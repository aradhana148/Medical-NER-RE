import spacy
import scispacy
import pandas as pd
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.matcher import Matcher

df = pd.read_csv("./final_gt.csv")

clean = (
    df.iloc[:, 0]
      .astype(str)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
      .reset_index(drop=True)
)

concepts = df.iloc[:, 1].reset_index(drop=True)
print("Sample concepts:")
print(concepts[0:3])
print("\nSample texts:")
print(clean[0:3])

# Load all three models for comprehensive entity extraction
print("\nLoading models...")
nlp_sci = spacy.load("en_core_sci_sm")        # General biomedical entities (anatomy, procedures, etc.)
nlp_diseases = spacy.load("en_ner_bc5cdr_md")  # Disease and chemical entities
nlp_meds = spacy.load("en_core_med7_lg")       # Medication entities (drugs, dosage, etc.)

# Add negation detection to the sci_sm model using clinical termset
ts = termset("en_clinical")
nlp_sci.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

# Also add negation detection to the disease model
nlp_diseases.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

# Create matcher for location modifiers
matcher = Matcher(nlp_sci.vocab)

# Define patterns for location modifiers (common in radiology)
location_patterns = [
    [{"LOWER": "left"}, {"LOWER": "sided", "OP": "?"}],
    [{"LOWER": "right"}, {"LOWER": "sided", "OP": "?"}],
    [{"LOWER": "bilateral"}],
    [{"LOWER": "bilaterally"}],
    [{"LOWER": "upper"}],
    [{"LOWER": "lower"}],
    [{"LOWER": "mid"}],
    [{"LOWER": "anterior"}],
    [{"LOWER": "posterior"}],
    [{"LOWER": "medial"}],
    [{"LOWER": "lateral"}],
    [{"LOWER": "basal"}],
    [{"LOWER": "basilar"}],
    [{"LOWER": "apical"}],
]
matcher.add("LOCATION_MODIFIER", location_patterns)

# List of location modifier keywords for entity association
LOCATION_KEYWORDS = {"left", "right", "bilateral", "bilaterally", "upper", "lower", 
                     "mid", "anterior", "posterior", "medial", "lateral", "basal", 
                     "basilar", "apical", "sided"}

# Observation/Status keywords - common in radiology for describing findings
OBSERVATION_KEYWORDS = {
    # Normal/clear findings
    "clear", "normal", "unremarkable", "stable", "unchanged", "intact", 
    "well-expanded", "well-aerated", "midline", "symmetric",
    # Abnormal findings
    "enlarged", "enlarged,", "calcified", "tortuous", "hyperinflated",
    "congestion", "opacity", "opacities", "effusion", "consolidation",
    "atelectasis", "edema", "thickening", "cardiomegaly", "pneumothorax",
    # Size descriptors
    "small", "moderate", "large", "mild", "minimal", "tiny", "top-normal"
}

print("Models loaded with negation detection and location modifiers!")

# Helper function to find location modifiers near an entity
def find_modifiers(doc, ent):
    """Find location modifiers in the context around an entity."""
    modifiers = []
    
    # Look at tokens before and after the entity (within a window)
    window_size = 5
    start_idx = max(0, ent.start - window_size)
    end_idx = min(len(doc), ent.end + window_size)
    
    for token in doc[start_idx:end_idx]:
        if token.text.lower() in LOCATION_KEYWORDS:
            modifiers.append(token.text.lower())
    
    # Also check if modifier is part of the entity itself
    for token in ent:
        if token.text.lower() in LOCATION_KEYWORDS:
            if token.text.lower() not in modifiers:
                modifiers.append(token.text.lower())
    
    return list(set(modifiers))  # Remove duplicates

# Helper function to find observations/status near an entity
def find_observations(doc, ent):
    """Find observation/status words in the context around an entity."""
    observations = []
    
    # Look at tokens before and after the entity (within a window)
    window_size = 5
    start_idx = max(0, ent.start - window_size)
    end_idx = min(len(doc), ent.end + window_size)
    
    for token in doc[start_idx:end_idx]:
        word = token.text.lower().rstrip(',.')
        if word in OBSERVATION_KEYWORDS:
            observations.append(word)
    
    # Also check if observation is part of the entity itself
    for token in ent:
        word = token.text.lower().rstrip(',.')
        if word in OBSERVATION_KEYWORDS:
            if word not in observations:
                observations.append(word)
    
    return list(set(observations))  # Remove duplicates

# This function generates annotations for each entity and label
def generate_annotation(texts):
    annotations = []
    for text in texts:
        doc_sci = nlp_sci(text)
        doc_diseases = nlp_diseases(text)
        doc_meds = nlp_meds(text)
        
        entities = []
        seen = set()  # Track (start, end) to avoid duplicates
        
        # Add general biomedical entities from sci_sm model (with negation)
        for ent in doc_sci.ents:
            key = (ent.start_char, ent.end_char)
            if key not in seen:
                seen.add(key)
                modifiers = find_modifiers(doc_sci, ent)
                observations = find_observations(doc_sci, ent)
                entities.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': 'ENTITY',
                    'text': ent.text,
                    'source': 'sci_sm',
                    'negated': ent._.negex,
                    'modifiers': modifiers,
                    'observations': observations
                })
        
        # Add disease/chemical entities from bc5cdr model (with negation)
        for ent in doc_diseases.ents:
            key = (ent.start_char, ent.end_char)
            if key not in seen:
                seen.add(key)
                modifiers = find_modifiers(doc_diseases, ent)
                observations = find_observations(doc_diseases, ent)
                entities.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                    'text': ent.text,
                    'source': 'bc5cdr',
                    'negated': ent._.negex,
                    'modifiers': modifiers,
                    'observations': observations
                })
        
        # Add medication entities from med7 model (no negation for this model)
        for ent in doc_meds.ents:
            key = (ent.start_char, ent.end_char)
            if key not in seen:
                seen.add(key)
                modifiers = find_modifiers(doc_meds, ent)
                observations = find_observations(doc_meds, ent)
                entities.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                    'text': ent.text,
                    'source': 'med7',
                    'negated': False,
                    'modifiers': modifiers,
                    'observations': observations
                })
        
        # Sort entities by position
        entities.sort(key=lambda x: x['start'])
        annotations.append((text, {'entities': entities}))
    return annotations

# ============================================================================
# POST-PROCESSING: Generate human-readable concepts from NER output
# ============================================================================

# Generic skip words (non-informative terms, not domain-specific)
SKIP_TERMS = {
    # View/imaging terms
    'pa', 'ap', 'frontal', 'lateral', 'views', 'radiographs', 'radiograph',
    'chest', 'view', 'film', 'image', 'study', 'exam', 'examination',
    'portable', 'single', 'obtained', 'provided', 'performed',
    # Generic terms
    'post', 'changes', 'noted', 'seen', 'detected', 'identified', 'demonstrated',
    'evidence', 'imaged', 'imaging', 'comparison', 'prior', 'status',
    'patient', 'finding', 'findings', 'appearance', 'appears', 'region', 'area',
    'side', 'aspect', 'level', 'position', 'expected', 'appropriate', 'interval',
    'limits', 'within', 'without', 'however', 'otherwise', 'there', 'are', 'is',
    # Standalone descriptors (should modify other terms)
    'low', 'bilaterally', 'stable', 'unchanged', 'unremarkable', 'intact',
    'clear', 'normal', 'mild', 'moderate', 'minimal', 'increased', 'decreased',
    'small', 'large', 'tiny', 'prominent', 'mildly', 'slightly', 'midline',
    # Generic anatomical terms too vague alone
    'contiguous', 'terminate', 'suggestive', 'interim', 'configuration',
    'silhouette', 'contour', 'contours', 'consistent', 'no free', 'free air'
}

# Observation words that can modify entities
OBSERVATION_WORDS = {
    'clear', 'normal', 'stable', 'unchanged', 'unremarkable', 'intact',
    'enlarged', 'calcified', 'tortuous', 'hyperinflated', 'mild', 'moderate',
    'severe', 'small', 'large', 'minimal', 'prominent', 'increased', 'decreased'
}

def is_valid_entity(text, label):
    """Check if entity should be kept based on generic rules."""
    text_lower = text.lower().strip()
    
    # Skip if in skip list
    if text_lower in SKIP_TERMS:
        return False
    
    # Skip very short entities (less than 3 chars)
    if len(text_lower) < 3:
        return False
    
    # Skip single character or number-only entities
    if text_lower.isdigit() or len(text_lower) == 1:
        return False
    
    # Keep entities with specific labels from disease/chemical model
    if label in {'DISEASE', 'CHEMICAL'}:
        return True
    
    # Keep multi-word entities (more descriptive)
    words = text_lower.split()
    if len(words) >= 2:
        # Skip if first word is just a determiner/preposition
        skip_first = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'no'}
        if words[0] not in skip_first:
            return True
    
    # For single words, keep if length > 4 (more likely to be meaningful)
    if len(text_lower) > 4:
        return True
    
    return False

def generate_concept(entity):
    """Generate a human-readable concept from an entity."""
    text = entity['text'].lower().strip()
    negated = entity['negated']
    label = entity.get('label', 'ENTITY')
    modifiers = entity.get('modifiers', [])
    observations = entity.get('observations', [])
    
    # Skip invalid entities
    if not is_valid_entity(text, label):
        return None
    
    # Skip if entity text matches an observation word (standalone)
    if text in OBSERVATION_WORDS:
        return None
    
    parts = []
    
    # Add negation prefix
    if negated:
        parts.append("no")
    
    # Add location for non-negated entities
    if not negated:
        location_terms = [m for m in modifiers if m in {'left', 'right', 'bilateral'}]
        if location_terms:
            parts.append(location_terms[0])
    
    # Add observation descriptor for non-negated entities
    if not negated:
        for obs in observations:
            if obs in OBSERVATION_WORDS and obs not in text:
                parts.append(obs)
                break
    
    # Add the main entity text
    parts.append(text)
    
    # Clean up
    concept = ' '.join(parts)
    words = concept.split()
    cleaned = []
    for word in words:
        if word not in cleaned and word not in {'sided', 'the', 'a', 'an', 'is', 'are'}:
            cleaned.append(word)
    
    return ' '.join(cleaned) if len(cleaned) > 0 else None

def concepts_are_similar(c1, c2):
    """Check if two concepts are semantically similar."""
    w1 = set(c1.lower().split())
    w2 = set(c2.lower().split())
    
    # Remove common stopwords
    stopwords = {'no', 'the', 'a', 'an', 'is', 'are', 'of', 'in', 'with', 'without'}
    w1 = w1 - stopwords
    w2 = w2 - stopwords
    
    if not w1 or not w2:
        return False
    
    # If one is subset of other, they're similar
    if w1.issubset(w2) or w2.issubset(w1):
        return True
    
    # High overlap means similar
    intersection = w1 & w2
    smaller = min(len(w1), len(w2))
    if smaller > 0 and len(intersection) / smaller >= 0.8:
        return True
    
    return False

def generate_concepts_from_entities(entities):
    """Generate a list of unique concepts from entities."""
    concepts = []
    
    for entity in entities:
        concept = generate_concept(entity)
        if concept and len(concept) > 3:
            # Check for duplicates
            is_duplicate = False
            for existing in concepts:
                if concepts_are_similar(concept, existing):
                    if len(concept) > len(existing):
                        concepts.remove(existing)
                        concepts.append(concept)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                concepts.append(concept)
    
    return concepts

# Extract text entities and labels from the dataset (transcription)
medical_doc = clean.tolist()

# Generate annotations
annotations = generate_annotation(medical_doc)

# ============================================================================
# F1 SCORE CALCULATION WITH FUZZY MATCHING
# ============================================================================

def similarity2(a: str, b: str) -> float:
    """Calculate token-based similarity between two strings."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())

    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    intersection = a_tokens & b_tokens
    return (2 * len(intersection)) / (len(a_tokens) + len(b_tokens))


def fuzzy_counts_one(gt_list, pred_list, threshold):
    """Calculate TP, FP, FN for one document using fuzzy matching."""
    gt = [g.strip().lower() for g in gt_list if g.strip()]
    pr = [p.strip().lower() for p in pred_list if p.strip()]

    used_gt = set()
    TP = 0

    for p in pr:
        best_j = None
        best_score = 0.0

        for j, g in enumerate(gt):
            if j in used_gt:
                continue

            score = similarity2(p, g)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None and best_score >= threshold:
            TP += 1
            used_gt.add(best_j)

    FP = len(pr) - TP
    FN = len(gt) - TP

    return TP, FP, FN


def fuzzy_prf(gt_norm, pred_norm, threshold):
    """Calculate Precision, Recall, F1 across all documents."""
    TP = FP = FN = 0

    for gt, pr in zip(gt_norm, pred_norm):
        t, f, n = fuzzy_counts_one(gt, pr, threshold)
        TP += t
        FP += f
        FN += n

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0

    return precision, recall, f1, (TP, FP, FN)


def parse_concept_list(concept_str):
    """Parse concept string from CSV into a list."""
    import ast
    try:
        # Try to parse as Python list literal
        return ast.literal_eval(concept_str)
    except:
        # Fallback: extract items from string
        return [c.strip().strip("'\"") for c in concept_str.strip("[]").split(",")]


# Collect ground truth and predicted concepts for all documents
print("\n" + "="*80)
print("COLLECTING CONCEPTS FOR F1 CALCULATION")
print("="*80)

gt_norm = []  # Ground truth concepts
pred_norm = []  # Predicted concepts from NER

num_docs = len(annotations)  # Process all 401 documents

for i in range(num_docs):
    # Parse ground truth concepts from CSV
    gt_concepts = parse_concept_list(concepts[i])
    gt_norm.append(gt_concepts)
    
    # Generate predicted concepts from NER
    pred_concepts = generate_concepts_from_entities(annotations[i][1]['entities'])
    pred_norm.append(pred_concepts)
    
    # Print first 10 for visual inspection
    if i in (3,4,38,13,24,27,67):
        print(f"\n--- Document {i+1} ---")
        print(f"GT:   {gt_concepts}")
        print(f"PRED: {pred_concepts}")
        
# Calculate F1 scores at different thresholds
print("\n" + "="*80)
print("F1 SCORE RESULTS (FUZZY MATCHING)")
print("="*80)
print(f"\nTotal documents evaluated: {num_docs}")
print(f"\nThreshold | Precision | Recall  | F1 Score | (TP, FP, FN)")
print("-" * 60)

for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
    P, R, F1, counts = fuzzy_prf(gt_norm, pred_norm, threshold=th)
    print(f"   {th:.2f}   |   {P:.4f}  | {R:.4f} |  {F1:.4f}  | {counts}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("""
- Precision: What fraction of predicted concepts match ground truth
- Recall: What fraction of ground truth concepts were predicted
- F1: Harmonic mean of precision and recall
- Higher threshold = stricter matching (requires more word overlap)
""")