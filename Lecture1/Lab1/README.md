README_Lab1.md — Building a High-Quality Dataset
# 🧠 DVRC Lecture 1 — Building a High-Quality Dataset

## 🎯 Objective
The goal of this lab is to understand how to **build, clean, and document a dataset** suitable for machine learning or data analysis.  
You will learn how to:
- Identify the characteristics of a high-quality dataset  
- Perform data cleaning and transformation  
- Add metadata and structure for reproducibility  
- Export your dataset in a usable format (CSV, Parquet, JSON)

---

## 🧰 Step 1. Characteristics of a High-Quality Dataset

| Criterion | Description | Example |
|------------|--------------|----------|
| **Completeness** | No missing values or placeholders | Fill NA with imputation or remove |
| **Consistency** | Same units and formats across features | e.g., “height” always in cm |
| **Accuracy** | Data reflects real-world facts | Validate using external sources |
| **Relevance** | Only keep features useful for the task | Drop irrelevant columns |
| **Traceability** | Each record has an identifiable origin | Use unique IDs |

---

## 🧼 Step 2. Data Cleaning Workflow

1. **Inspect your dataset**
   ```python
   import pandas as pd
   df = pd.read_csv('data/raw_dataset.csv')
   df.info()
   df.describe()

    ```

2.  **Handle missing values**
```python
df = df.dropna(subset=['target'])  # or
df['age'].fillna(df['age'].mean(), inplace=True)
```

3.  **Normalize formats**
```python
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
```

4. **Detect duplicates**
```python
    df = df.drop_duplicates()
```

5. **Export the cleaned version**
```python
    df.to_csv('data/clean_dataset.csv', index=False)
```

## 📜 Step 3. Document your dataset (Metadata)

Create a README_DATASET.md inside /data/ with:

    - Source (where it comes from)
    - License (open, academic, private)
    - Columns description
    - Date of creation
    - Preprocessing steps
    - Author(s)

Example:

## Dataset Documentation

**Source:** Kaggle - Superstore Sales  
**License:** CC BY 4.0  

**Columns:**
- `order_id`: unique identifier
- `sales`: total sales amount
- `profit`: profit margin (%)

## 🚀 Step 4. Best Practices

    - Always keep raw/ data untouched.
    - Work inside data/interim/ for cleaning.
    - Final datasets go to data/processed/.
    - Use version control (git) for scripts, not for large data.
    - Use .gitignore for large .csv files (>50MB).

### **Example structure:**

    project/
    │
    ├── data/
    │   ├── raw/
    │   ├── interim/
    │   └── processed/
    ├── notebooks/
    ├── src/
    ├── requirements.txt
    └── README.md

## 🎓 Deliverables

- data/clean_dataset.csv
- README_DATASET.md describing it
- Short notebook showing cleaning & EDA steps

## 🧠 Evaluation Criteria
* Criterion	Weight
* Data quality & cleaning	40%
* Documentation	30%
* Structure & reproducibility	20%
* Presentation (clarity)	10%