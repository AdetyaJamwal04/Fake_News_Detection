import spacy

nlp = spacy.load("en_core_web_sm")

def generate_queries(claim: str):
    """
    Generate multiple search queries from a claim.
    """
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents]

    base = claim.lower()

    queries = [
        base,
        base + " fact check",
        base + " true or false",
        base + " hoax",
        base + " authenticity check",
    ]

    for e in entities:
        queries.append(f"{e} {base}")
        queries.append(f"{base} {e} false")
        queries.append(f"{e} controversy")
        queries.append(f"{e} news verification")

    return list(set(queries))
