# Association Analysis — Conversation Log

**Date:** November 5, 2026  
**Topic:** Association rule mining, redundancy, rule lattices, hypercliques, and cross-support filtering using Pandas, NumPy, scikit-learn, and MLxtend.

This is an organized record of the user's prompts and the assistant's answers. Explanations and code have been consolidated for readability rather than reproduced as a verbatim transcript.

---

## Exchange 1 — Extracting association rules

**User prompt:**
> while using Pandas, NumPy, Scikit-learn, and MLxtend how to extract association rules with confidence and lift thresholds for the online retail and groceries Benchmark datasets

**Assistant answer:**

The pipeline is (1) load and clean transactions, (2) one-hot encode invoice/item baskets, (3) mine frequent itemsets with Apriori or FP-Growth, (4) generate association rules using confidence, and (5) filter by lift.

Metrics for A → B:

- Support: P(A ∩ B)
- Confidence: P(B | A) = support(A ∪ B) / support(A)
- Lift: confidence(A → B) / support(B)

For Online Retail, discard cancelled invoices (InvoiceNo starting with `C`), nonpositive quantities, and missing invoice/product identifiers. Use InvoiceNo as transaction ID and StockCode as item ID. For Groceries, load comma-separated transactions and use `TransactionEncoder` to build a Boolean basket matrix.

```python
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

retail = pd.read_excel('Online Retail.xlsx')
retail = retail.dropna(subset=['InvoiceNo', 'StockCode'])
retail = retail[~retail['InvoiceNo'].astype(str).str.startswith('C')]
retail = retail[retail['Quantity'] > 0].copy()
retail['StockCode'] = retail['StockCode'].astype(str)
retail_basket = pd.crosstab(retail['InvoiceNo'], retail['StockCode']).gt(0)

with open('groceries.csv', encoding='utf-8') as f:
    transactions = [
        [item.strip() for item in line.strip().split(',') if item.strip()]
        for line in f if line.strip()
    ]
encoder = TransactionEncoder()
arr = encoder.fit(transactions).transform(transactions, sparse=True)
groceries_basket = pd.DataFrame.sparse.from_spmatrix(
    arr, columns=encoder.columns_
).astype(pd.SparseDtype(bool, False))
```

A reusable extraction function:

```python
def extract_rules(basket, min_support=0.01,
                  min_confidence=0.60, min_lift=1.20,
                  algorithm='fpgrowth'):
    if algorithm == 'apriori':
        itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    elif algorithm == 'fpgrowth':
        itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    else:
        raise ValueError('Unknown algorithm')
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric='confidence',
                              min_threshold=min_confidence,
                              num_itemsets=len(basket))
    rules = rules[(rules['confidence'] >= min_confidence) &
                  (rules['lift'] >= min_lift)].copy()
    return rules.sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True)
```

Example thresholds: Online Retail support 0.01, confidence 0.60, lift 1.20; Groceries support 0.005, confidence 0.30, lift 1.20. Thresholds are illustrative and should be tuned. Benchmark Apriori versus FP-Growth by execution time and number of rules. For hypercliques, all-confidence is also required.

---

## Exchange 2 — Generating rules

**User prompt:**
> how do i generate the rules?

**Assistant answer:**

First mine frequent itemsets, then pass them into `association_rules`. `metric='confidence'` enforces the confidence threshold; lift is filtered afterward.

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules

frequent_itemsets = fpgrowth(retail_basket, min_support=0.01,
                            use_colnames=True)

rules = association_rules(frequent_itemsets,
                          metric='confidence',
                          min_threshold=0.60,
                          num_itemsets=len(retail_basket))

filtered_rules = rules[(rules['confidence'] >= 0.60) &
                       (rules['lift'] >= 1.20)].copy()
filtered_rules = filtered_rules.sort_values('lift', ascending=False)

print(filtered_rules[['antecedents', 'consequents',
                      'support', 'confidence', 'lift']].head(10))
```

For Groceries, substitute `groceries_basket`, typically using a smaller support threshold. If no rules appear, lower the support and/or confidence thresholds. In newer MLxtend versions, supply `num_itemsets`.

---

## Exchange 3 — Removing redundant rules

**User prompt:**
> how to remove the reduntant rules?

**Assistant answer:**

A confidence-based redundancy criterion: with the same consequent Y, a more specific rule X → Y can be removed if a simpler antecedent A is a strict subset of X and confidence(A → Y) ≥ confidence(X → Y).

Example: Milk → Bread (0.80 confidence) and Milk, Butter → Bread (0.80). The second rule adds no confidence improvement.

```python
def remove_redundant_rules(rules, tolerance=1e-9):
    rules = rules.copy().reset_index(drop=True)
    rules = rules.drop_duplicates(
        subset=['antecedents', 'consequents']
    ).reset_index(drop=True)
    redundant = set()
    for i, ri in rules.iterrows():
        for j, rj in rules.iterrows():
            if i == j:
                continue
            if (ri['consequents'] == rj['consequents'] and
                rj['antecedents'] < ri['antecedents'] and
                rj['confidence'] >= ri['confidence'] - tolerance):
                redundant.add(i)
                break
    return rules.drop(index=list(redundant)).reset_index(drop=True)

retail_clean = remove_redundant_rules(retail_rules)
groceries_clean = remove_redundant_rules(groceries_rules)
```

Report before/after rule counts and percent reduction. Redundancy has multiple formal definitions, including improvement-based and closed-itemset pruning. Filtering rules before comparing can hide simpler candidate rules; for rigorous pruning, compare against a sufficiently broad rule set.

---

## Exchange 4 — Constructing rule lattices

**User prompt:**
> how to construct the rle lattices?

**Assistant answer:**

Distinguish frequent-itemset lattices (nodes are itemsets) from association-rule hierarchies (nodes are rules). A fixed-consequent rule hierarchy connects rules when their antecedents are in an immediate subset relationship.

Example hierarchy:

```text
         {A} → {D}
          /      \
 {A,B} → {D}   {A,C} → {D}
          \      /
        {A,B,C} → {D}
```

Construct a directed graph using NetworkX. Nodes are `(frozenset(antecedent), frozenset(consequent))`. Add edges from smaller to larger antecedents with identical consequents, omitting edges when an intermediate rule exists.

```python
import networkx as nx
import matplotlib.pyplot as plt

def construct_rule_lattice(rules, consequent=None):
    G = nx.DiGraph()
    selected = rules.copy()
    if consequent is not None:
        target = frozenset(consequent)
        selected = selected[selected['consequents'] == target]
    for _, rule in selected.iterrows():
        node = (frozenset(rule['antecedents']),
                frozenset(rule['consequents']))
        G.add_node(node, confidence=rule['confidence'],
                   lift=rule['lift'], support=rule['support'])
    nodes = list(G.nodes())
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            ant_a, cons_a = a
            ant_b, cons_b = b
            if cons_a == cons_b and ant_a < ant_b:
                intermediate = any(
                    cons_c == cons_a and ant_a < ant_c < ant_b
                    for ant_c, cons_c in nodes
                )
                if not intermediate:
                    G.add_edge(a, b)
    return G
```

Select a frequent consequent and a manageable subset of rules, then draw with `nx.draw` or a layered layout. For redundancy, mark nodes red if any more general ancestor has at least as high confidence:

```python
def find_redundant_rules(G, tolerance=1e-9):
    redundant = []
    for node in G.nodes():
        conf = G.nodes[node]['confidence']
        if any(G.nodes[a]['confidence'] >= conf - tolerance
               for a in nx.ancestors(G, node)):
            redundant.append(node)
    return redundant
```

This is a Hasse diagram of the **available fixed-consequent rules**, not necessarily a complete mathematical lattice. Construct it before pruning to show which nodes are removed.

---

## Exchange 5 — Lift, cosine, and all-confidence for hypercliques

**User prompt:**
> how to use lift, cosine, and all confidence to identify hypercliques

**Assistant answer:**

For two items A and B:

- Lift(A,B) = support(A,B) / [support(A) × support(B)]
- Cosine(A,B) = support(A,B) / sqrt[support(A) × support(B)]
- All-confidence(X) = support(X) / max_{i in X} support({i})

**Hyperclique definition:** An itemset X is a hyperclique when all-confidence(X) ≥ h, for the chosen h-confidence threshold. Lift and cosine are additional descriptive/screening measures, not part of the formal definition.

Example: support(A)=0.80, support(B)=0.10, support(A,B)=0.09 gives lift=1.125, cosine≈0.318, all-confidence=0.1125. It fails a hyperclique threshold of 0.60.

Calculate metrics directly on frequent itemsets (including those with 3+ items):

```python
import numpy as np
import pandas as pd

def calculate_hyperclique_metrics(frequent_itemsets):
    singleton_support = {
        next(iter(items)): support
        for items, support in zip(frequent_itemsets['itemsets'],
                                  frequent_itemsets['support'])
        if len(items) == 1
    }
    rows = []
    for _, row in frequent_itemsets.iterrows():
        items = row['itemsets']
        if len(items) < 2:
            continue
        support = row['support']
        supports = np.array([singleton_support[i] for i in items])
        k = len(items)
        rows.append({
            'itemset': items,
            'size': k,
            'support': support,
            'lift': support / np.prod(supports),
            'cosine': support / (np.prod(supports) ** (1 / k)),
            'all_confidence': support / max(supports)
        })
    return pd.DataFrame(rows)

itemsets = fpgrowth(groceries_basket, min_support=0.005,
                    use_colnames=True)
metrics_df = calculate_hyperclique_metrics(itemsets)
hypercliques = metrics_df[
    metrics_df['all_confidence'] >= 0.60
].sort_values('all_confidence', ascending=False)
```

For larger itemsets, the cosine and lift calculations above are generalized versions; they are not directional association-rule metrics. An optional stronger screen is all-confidence ≥ 0.60, cosine ≥ 0.50, lift ≥ 1.20.

All-confidence is anti-monotone: X ⊆ Y implies hconf(Y) ≤ hconf(X). Thus, supersets of a failing itemset can be pruned during mining. Benchmark hyperclique counts over multiple h-confidence thresholds.

---

## Exchange 6 — Eliminating cross-support spurious rules

**User prompt:**
> how to eliminate cross-support spurios rules?

**Assistant answer:**

Cross-support patterns combine items with highly unequal singleton supports. Define:

CSR(X) = min_{i in X} support({i}) / max_{i in X} support({i})

All-confidence satisfies hconf(X) ≤ CSR(X). Therefore, if CSR(X) < h, X cannot be a hyperclique at threshold h. This is the cross-support pruning property.

Example: support(A)=0.80, support(B)=0.05, support(A,B)=0.045. Confidence(B → A)=0.90, but confidence(A → B)=0.05625 and CSR=0.0625. High directional confidence alone can be misleading.

```python
def cross_support_metrics(itemset, support, singleton_support):
    values = [singleton_support[i] for i in itemset]
    return min(values) / max(values), support / max(values)


def analyze_cross_support(rules, frequent_itemsets, h=0.6):
    singletons = {
        next(iter(items)): support
        for items, support in zip(frequent_itemsets['itemsets'],
                                  frequent_itemsets['support'])
        if len(items) == 1
    }
    df = rules.copy()
    metrics = [
        cross_support_metrics(
            row['antecedents'] | row['consequents'],
            row['support'], singletons
        )
        for _, row in df.iterrows()
    ]
    df['CSR'] = [m[0] for m in metrics]
    df['hconf'] = [m[1] for m in metrics]
    df['classification'] = np.select(
        [df['CSR'] < h, df['hconf'] < h],
        ['Cross-support rejected', 'Low h-confidence rejected'],
        default='Hyperclique'
    )
    return df

retail_analysis = analyze_cross_support(
    retail_rules, retail_itemsets, h=0.60
)
groceries_analysis = analyze_cross_support(
    groceries_rules, groceries_itemsets, h=0.60
)

print(retail_analysis['classification'].value_counts())
print(groceries_analysis['classification'].value_counts())
```

A separate CSR filter may be used for diagnostics. If the CSR threshold is below the h-confidence threshold, it does not further change which itemsets pass the h-confidence requirement. Report counts and percentages of cross-support rejections, low-h-confidence rejections, and retained hypercliques. Cross-support filtering is a chosen pattern-quality criterion, **not proof that a rule is statistically false**.

---

## Exchange 7 — Requesting a conversation log

**User prompt:**
> could you make a log of this conversation including my prompts and your answers

**Assistant answer:**

This document is the requested organized conversation log, covering the six preceding technical questions, their explanations, formulas, and representative implementation code. It is a consolidated study/project record, not a word-for-word export of the chat.

---

## Implementation notes

- `retail_basket` and `groceries_basket` must be Boolean transaction-by-item matrices.
- `retail_itemsets` and `groceries_itemsets` must correspond to the respective dataset and mining support threshold.
- `association_rules(..., num_itemsets=len(basket))` is suitable for recent MLxtend releases.
- Pruning results depend on the candidate rules retained before pruning.
- All-confidence is the primary hyperclique definition; lift and cosine supplement interpretation.
- For a project report, document the exact thresholds, preprocessing, rule counts, runtime, and percentage of rules removed.
