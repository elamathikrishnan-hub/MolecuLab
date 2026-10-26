# 🧪 MolecuLab

**Interactive Cheminformatics Virtual Laboratory**

MolecuLab is an educational web app for exploring molecular structures and
physicochemical descriptors. It studies three molecules: **Ibuprofen**,
**Nicotine** and **Quercetin**.



## Features
- **Theory:** short introduction to cheminformatics, SMILES and descriptors
- **Molecular Visualization:** 2D structures and interactive 3D models
  (rotate and zoom), plus a SMILES validator for your own molecules
- **Molecular Properties Report:** formula, molecular weight, LogP, TPSA,
  H-bond donors/acceptors, atom and bond counts, with CSV download
- **Structure–Property Analysis:** compare descriptors across molecules
- **Assessment:** five-question quiz with explanations
- **Molecules Report:** downloadable report describing the three molecules and their properties

## Built with
Python · Streamlit · RDKit · py3Dmol · pandas

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System libraries needed by RDKit on Streamlit Cloud |

## Note
Descriptors are calculated values for learning and comparison. They do not by
themselves establish a molecule's safety or effectiveness.

## References
- RDKit documentation: https://www.rdkit.org/docs/
- Streamlit documentation: https://docs.streamlit.io/
- py3Dmol: https://pypi.org/project/py3Dmol/
