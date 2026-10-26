
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from io import BytesIO

st.set_page_config(
    page_title="MolecuLab",
    page_icon="🧪",
    layout="wide"
)

MOLECULES = {
    "Ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "description": "A nonsteroidal anti-inflammatory drug (NSAID)."
    },
    "Nicotine": {
        "smiles": "CN1CCC[C@H]1c2cccnc2",
        "description": "An alkaloid found naturally in tobacco plants."
    },
    "Quercetin": {
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "description": "A plant flavonoid found in many fruits and vegetables."
    }
}

def get_molecule(name):
    mol = Chem.MolFromSmiles(MOLECULES[name]["smiles"])
    if mol is None:
        return None
    return mol

def get_properties(name):
    mol = get_molecule(name)
    return {
        "Molecule": name,
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Bonds": mol.GetNumBonds()
    }

st.sidebar.title("🧪 MolecuLab")
st.sidebar.caption("Interactive Cheminformatics Virtual Laboratory")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Theory",
        "Molecular Visualization",
        "Molecular Properties Report",
        "Structure–Property Analysis",
        "Assessment",
        "Final Project Report"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Student Information")
student_name = st.sidebar.text_input("Student Name")
registration = st.sidebar.text_input("Registration Number")

if page == "Home":
    st.title("🧪 MolecuLab")
    st.subheader("Interactive Cheminformatics Virtual Laboratory")
    st.write(
        "Explore molecular structures, calculate chemical properties, "
        "and compare important molecular descriptors."
    )
    st.info(
        "This educational application explores three molecules: "
        "Ibuprofen, Nicotine, and Quercetin."
    )

    cols = st.columns(3)
    for col, name in zip(cols, MOLECULES):
        with col:
            st.markdown(f"### {name}")
            st.write(MOLECULES[name]["description"])
            st.caption(f"SMILES: {MOLECULES[name]['smiles']}")

    st.markdown("### What can you explore?")
    st.write(
        "- Visualize molecular structures\n"
        "- Explore molecular properties\n"
        "- Compare structures and descriptors\n"
        "- Take a chemistry assessment\n"
        "- Download your project report"
    )

elif page == "Theory":
    st.title("Theory of Cheminformatics")
    st.subheader("What is cheminformatics?")
    st.write(
        "Cheminformatics combines chemistry, computer science, and data "
        "analysis to store, represent, visualize, and study chemical molecules."
    )
    st.subheader("What is SMILES?")
    st.write(
        "SMILES stands for Simplified Molecular Input Line Entry System. "
        "It represents a molecular structure using text characters."
    )
    st.subheader("Molecular descriptors")
    st.write(
        "- Molecular weight: mass of a molecule in g/mol.\n"
        "- LogP: estimates partitioning between octanol and water.\n"
        "- TPSA: estimates the topological polar surface area.\n"
        "- H-bond donors and acceptors: describe hydrogen-bonding features."
    )

elif page == "Molecular Visualization":
    st.title("Molecular Visualization")
    name = st.selectbox("Select a molecule", list(MOLECULES))
    mol = get_molecule(name)

    st.success(f"{name} loaded successfully.")
    st.write("**SMILES notation:**", MOLECULES[name]["smiles"])
    st.write("**Molecular formula:**", rdMolDescriptors.CalcMolFormula(mol))

    image = Draw.MolToImage(mol, size=(600, 400))
    st.image(image, caption=f"2D structure of {name}")

    st.caption(
        "This version displays 2D structures. Interactive 3D visualization "
        "can be added in a later step."
    )

    st.subheader("Structure information")
    st.write("Heavy atoms:", mol.GetNumHeavyAtoms())
    st.write("Total atoms:", mol.GetNumAtoms())
    st.write("Bonds:", mol.GetNumBonds())

    st.subheader("Enter your own SMILES")
    user_smiles = st.text_input("SMILES notation")
    if st.button("Validate SMILES"):
        custom_mol = Chem.MolFromSmiles(user_smiles)
        if custom_mol:
            st.success("Valid SMILES notation.")
            st.image(Draw.MolToImage(custom_mol, size=(500, 350)))
            st.write(
                "Molecular formula:",
                rdMolDescriptors.CalcMolFormula(custom_mol)
            )
        else:
            st.error("Invalid SMILES notation. Please check your input.")

elif page == "Molecular Properties Report":
    st.title("Molecular Properties Report")
    selected = st.selectbox("Choose a molecule", list(MOLECULES))
    properties = get_properties(selected)

    for key, value in properties.items():
        if key != "Molecule":
            st.metric(key, value)

    st.subheader("All three molecules")
    df = pd.DataFrame([get_properties(n) for n in MOLECULES])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download properties as CSV",
        csv,
        "moleculab_properties.csv",
        "text/csv"
    )

elif page == "Structure–Property Analysis":
    st.title("Structure–Property Analysis")
    df = pd.DataFrame([get_properties(n) for n in MOLECULES])
    st.dataframe(df, use_container_width=True)

    numeric_columns = [
        "Molecular Weight (g/mol)",
        "LogP",
        "TPSA (Å²)",
        "H-Bond Donors",
        "H-Bond Acceptors"
    ]
    choice = st.selectbox("Choose a property to compare", numeric_columns)

    chart = df.set_index("Molecule")[choice]
    st.bar_chart(chart)

    st.subheader("Interpretation")
    st.write(
        "Different molecular structures produce different physicochemical "
        "properties. These descriptors help researchers compare molecules, "
        "but they do not by themselves establish safety or effectiveness."
    )

elif page == "Assessment":
    st.title("Chemistry Assessment")

    questions = [
        (
            "What does SMILES represent?",
            ["A molecular text notation", "A microscope", "A type of cell"],
            0,
            "SMILES encodes molecular structures as text."
        ),
        (
            "Which descriptor represents molecular mass?",
            ["LogP", "Molecular weight", "TPSA"],
            1,
            "Molecular weight is expressed in g/mol."
        ),
        (
            "What does TPSA describe?",
            ["Polar surface area", "Molecule color", "Boiling time"],
            0,
            "TPSA estimates the topological polar surface area."
        ),
        (
            "Which molecule is a plant flavonoid?",
            ["Nicotine", "Ibuprofen", "Quercetin"],
            2,
            "Quercetin is a flavonoid found in many plants."
        ),
        (
            "What does LogP help estimate?",
            ["Molecular age", "Partitioning between octanol and water",
             "Number of atoms"],
            1,
            "LogP estimates a molecule's relative lipophilicity."
        )
    ]

    with st.form("quiz"):
        answers = []
        for i, (question, options, correct, explanation) in enumerate(questions):
            answer = st.radio(
                f"{i + 1}. {question}",
                options,
                index=None,
                key=f"q{i}"
            )
            answers.append(answer)

        submitted = st.form_submit_button("Submit Assessment")

    if submitted:
        score = 0
        for answer, (question, options, correct, explanation) in zip(
            answers, questions
        ):
            if answer == options[correct]:
                score += 1

        st.success(f"Your score: {score}/{len(questions)}")

        for i, (question, options, correct, explanation) in enumerate(questions):
            st.write(f"**Question {i + 1}:** {explanation}")
            st.write(f"Correct answer: {options[correct]}")

elif page == "Final Project Report":
    st.title("Final Project Report")

    report = f"""MOLECULAB
Interactive Cheminformatics Virtual Laboratory

Student: {student_name or 'Not provided'}
Registration Number: {registration or 'Not provided'}

ABSTRACT
MolecuLab is an educational application for exploring molecular structures
and physicochemical descriptors using cheminformatics tools.

OBJECTIVES
1. Explore molecular representations and SMILES.
2. Visualize selected molecular structures.
3. Calculate and compare molecular descriptors.
4. Assess understanding through a chemistry quiz.

METHODOLOGY
Python, Streamlit, and RDKit are used to represent molecules and calculate
their physicochemical descriptors.

MOLECULES STUDIED
Ibuprofen, Nicotine, and Quercetin.

CONCLUSION
MolecuLab demonstrates how computational tools can support the study of
molecular structures and chemical properties.

REFERENCES
RDKit documentation: https://www.rdkit.org/docs/
Streamlit documentation: https://docs.streamlit.io/
"""

    st.text_area("Report preview", report, height=350)
    st.download_button(
        "Download project report",
        report.encode("utf-8"),
        "MolecuLab_Project_Report.txt",
        "text/plain"
    )

st.sidebar.divider()
st.sidebar.caption("MolecuLab | Educational Chemistry Project")