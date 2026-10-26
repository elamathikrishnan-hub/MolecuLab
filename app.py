import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw

st.set_page_config(
    page_title="MolecuLab",
    page_icon="🧪",
    layout="wide"
)

MOLECULES = {
    "Ibuprofen": {
        # Ibuprofen is sold as a racemate, so no stereocentre is specified
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
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
    return Chem.MolFromSmiles(MOLECULES[name]["smiles"])


@st.cache_data
def smiles_to_molblock(smiles):
    """Build a 3D structure (with hydrogens) and return it as a MolBlock."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


def show_3d(smiles, key):
    """Render an interactive 3D viewer for a SMILES string."""
    style_name = st.selectbox(
        "3D style", ["Ball and stick", "Stick", "Sphere"], key=key
    )
    styles = {
        "Ball and stick": {"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}},
        "Stick": {"stick": {}},
        "Sphere": {"sphere": {}},
    }
    molblock = smiles_to_molblock(smiles)
    if molblock is None:
        st.error("Could not generate a 3D structure for this molecule.")
        return
    view = py3Dmol.view(width=600, height=420)
    view.addModel(molblock, "mol")
    view.setStyle({}, styles[style_name])
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=440)
    st.caption("Drag to rotate. Pinch or scroll to zoom.")


def count_all_atoms(mol):
    """Total atoms including hydrogens (GetNumAtoms() ignores implicit H)."""
    return Chem.AddHs(mol).GetNumAtoms()


def count_all_bonds(mol):
    """Total bonds including bonds to hydrogens."""
    return Chem.AddHs(mol).GetNumBonds()


@st.cache_data
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
        "Total Atoms (with H)": count_all_atoms(mol),
        "Bonds (with H)": count_all_bonds(mol),
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
        "Molecules Report"
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
        "- Download a report on the molecules"
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

    if mol is None:
        st.error(f"Could not load {name}.")
    else:
        st.success(f"{name} loaded successfully.")
        st.write("**SMILES notation:**", MOLECULES[name]["smiles"])
        st.write("**Molecular formula:**", rdMolDescriptors.CalcMolFormula(mol))

        view_mode = st.radio("View mode", ["2D", "3D"], horizontal=True)
        if view_mode == "2D":
            image = Draw.MolToImage(mol, size=(600, 400))
            st.image(image, caption=f"2D structure of {name}")
        else:
            show_3d(MOLECULES[name]["smiles"], key="style_lib")

        st.subheader("Structure information")
        st.write("Heavy atoms:", mol.GetNumHeavyAtoms())
        st.write("Total atoms (including hydrogens):", count_all_atoms(mol))
        st.write("Total bonds (including hydrogens):", count_all_bonds(mol))

    st.subheader("Enter your own SMILES")
    user_smiles = st.text_input("SMILES notation")
    if st.button("Validate SMILES"):
        user_smiles = user_smiles.strip()
        if not user_smiles:
            st.session_state.pop("custom_smiles", None)
            st.warning("Please enter a SMILES string first.")
        elif Chem.MolFromSmiles(user_smiles) is None:
            st.session_state.pop("custom_smiles", None)
            st.error("Invalid SMILES notation. Please check your input.")
        else:
            st.session_state["custom_smiles"] = user_smiles

    # Kept in session_state so the 3D style menu works after validation
    if "custom_smiles" in st.session_state:
        custom_smiles = st.session_state["custom_smiles"]
        custom_mol = Chem.MolFromSmiles(custom_smiles)
        st.success("Valid SMILES notation.")
        st.image(Draw.MolToImage(custom_mol, size=(500, 350)))
        st.write(
            "Molecular formula:",
            rdMolDescriptors.CalcMolFormula(custom_mol)
        )
        st.subheader("3D structure")
        show_3d(custom_smiles, key="style_custom")

elif page == "Molecular Properties Report":
    st.title("Molecular Properties Report")
    selected = st.selectbox("Choose a molecule", list(MOLECULES))
    properties = get_properties(selected)

    for key, value in properties.items():
        if key != "Molecule":
            st.metric(key, value)

    st.subheader("All three molecules")
    df = pd.DataFrame([get_properties(n) for n in MOLECULES])
    st.dataframe(df, width="stretch")

    # utf-8-sig so Excel displays characters such as Å² correctly
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download properties as CSV",
        csv,
        "moleculab_properties.csv",
        "text/csv"
    )

elif page == "Structure–Property Analysis":
    st.title("Structure–Property Analysis")
    df = pd.DataFrame([get_properties(n) for n in MOLECULES])
    st.dataframe(df, width="stretch")

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
        if any(a is None for a in answers):
            st.warning("Please answer all questions before submitting.")
        else:
            score = sum(
                answer == options[correct]
                for answer, (_, options, correct, _) in zip(answers, questions)
            )
            st.success(f"Your score: {score}/{len(questions)}")

            for i, (question, options, correct, explanation) in enumerate(questions):
                st.write(f"**Question {i + 1}:** {explanation}")
                st.write(f"Correct answer: {options[correct]}")

elif page == "Molecules Report":
    st.title("Molecules Report")

    results_df = pd.DataFrame([get_properties(n) for n in MOLECULES])
    results_table = results_df[
        ["Molecule", "Formula", "Molecular Weight (g/mol)", "LogP",
         "TPSA (Å²)", "H-Bond Donors", "H-Bond Acceptors"]
    ].to_string(index=False)

    def mol_line(name):
        p = get_properties(name)
        return (
            f"SMILES: {MOLECULES[name]['smiles']}\n"
            f"Formula: {p['Formula']}\n"
            f"Molecular weight: {p['Molecular Weight (g/mol)']} g/mol\n"
            f"LogP: {p['LogP']}   TPSA: {p['TPSA (Å²)']} Å²\n"
            f"H-bond donors: {p['H-Bond Donors']}   "
            f"H-bond acceptors: {p['H-Bond Acceptors']}\n"
            f"Heavy atoms: {p['Heavy Atoms']}   "
            f"Total atoms (with H): {p['Total Atoms (with H)']}"
        )

    report = f"""STUDY OF IBUPROFEN, NICOTINE AND QUERCETIN
A Comparative Molecular Report

Student: {student_name or 'Not provided'}
Registration Number: {registration or 'Not provided'}

1. INTRODUCTION
Ibuprofen, nicotine and quercetin are three chemically different molecules
of biological and pharmaceutical interest. This report describes their
structures and compares their calculated physicochemical properties
(molecular descriptors).

2. IBUPROFEN
Class: nonsteroidal anti-inflammatory drug (NSAID), propionic acid derivative.
Chemical name: 2-(4-isobutylphenyl)propanoic acid.
{mol_line('Ibuprofen')}
Structure: a benzene ring carrying an isobutyl group and a propanoic acid
group. The carboxylic acid (-COOH) is the main polar group and the only
hydrogen-bond donor. The molecule has one stereocentre and is commonly sold
as a racemic mixture.
Use: relief of pain, fever and inflammation. It works by inhibiting
cyclooxygenase (COX) enzymes.

3. NICOTINE
Class: alkaloid found naturally in tobacco plants.
Chemical name: (S)-3-(1-methylpyrrolidin-2-yl)pyridine.
{mol_line('Nicotine')}
Structure: a pyridine ring joined to an N-methylpyrrolidine ring. It has two
nitrogen atoms, no hydroxyl or N-H groups, and therefore no hydrogen-bond
donors. The natural form is the (S)-enantiomer.
Biological role: it acts on nicotinic acetylcholine receptors and is the
addictive component of tobacco products.

4. QUERCETIN
Class: flavonoid (flavonol), a plant polyphenol.
{mol_line('Quercetin')}
Structure: a three-ring flavonol skeleton with five hydroxyl groups and a
ketone group. The rings are largely planar and conjugated.
Occurrence: found in many fruits and vegetables such as onions, apples and
berries. It shows antioxidant activity in laboratory studies.

5. COMPARISON OF CALCULATED PROPERTIES
{results_table}

6. DISCUSSION
Quercetin has the highest molecular weight, the largest topological polar
surface area and the most hydrogen-bond donors, which follows from its five
hydroxyl groups. Nicotine is the smallest of the three, with the lowest TPSA
and no hydrogen-bond donors. Ibuprofen is intermediate, with one donor (its
carboxylic acid) and a hydrophobic isobutylphenyl part. LogP and TPSA
together describe how lipophilic or polar each molecule is, which influences
solubility and membrane permeability. These values are computational
estimates and do not by themselves establish a molecule's safety,
effectiveness or biological activity.

7. METHOD
Structures were entered as SMILES strings. Molecular formula, molecular
weight, LogP, TPSA and hydrogen-bond counts were calculated with RDKit.
Atom counts include hydrogens.

8. CONCLUSION
Although all three molecules contain aromatic rings, differences in
functional groups lead to clearly different size, polarity and hydrogen-
bonding properties.

REFERENCES
RDKit documentation: https://www.rdkit.org/docs/
PubChem (National Library of Medicine): https://pubchem.ncbi.nlm.nih.gov/
"""

    st.text_area("Report preview", report, height=500)
    st.download_button(
        "Download molecules report",
        report.encode("utf-8"),
        "Molecules_Report.txt",
        "text/plain"
    )

st.sidebar.divider()
st.sidebar.caption("MolecuLab | Educational Chemistry Project")
