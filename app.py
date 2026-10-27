import re
from urllib.parse import quote
from xml.sax.saxutils import escape

import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MolecuLab | Cheminformatics Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F8FC;
        color: #172B4D;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DCE5F0;
    }

    section[data-testid="stSidebar"] * {
        color: #172B4D !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] label span {
        color: #172B4D !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    h1, h2, h3 {
        color: #174A8B;
    }

    .hero {
        background: linear-gradient(120deg, #174A8B, #3282CE);
        padding: 30px;
        border-radius: 16px;
        color: white !important;
        margin-bottom: 24px;
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .info-card {
        background-color: white;
        border: 1px solid #DCE5F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .small-note {
        color: #52657D;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DCE5F0;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] label {
        color: #52657D !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #174A8B !important;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MOLECULE DATABASE
# --------------------------------------------------

MOLECULES = {
    "Ibuprofen": {
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "formula": "C13H18O2",
        "description": (
            "Ibuprofen is a nonsteroidal anti-inflammatory drug "
            "commonly used to relieve pain, fever, and inflammation."
        ),
        "category": "Pharmaceutical",
    },
    "Nicotine": {
        "smiles": "CN1CCC[C@H]1C2=CN=CC=C2",
        "formula": "C10H14N2",
        "description": (
            "Nicotine is an alkaloid found naturally in tobacco "
            "plants. It acts on nicotinic acetylcholine receptors."
        ),
        "category": "Alkaloid",
    },
    "Quercetin": {
        "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(O2)C=C(C=C3O)O)O)O)O",
        "formula": "C15H10O7",
        "description": (
            "Quercetin is a plant flavonoid found in various fruits "
            "and vegetables. It is studied for its antioxidant properties."
        ),
        "category": "Flavonoid",
    },
}


# --------------------------------------------------
# MOLECULE FUNCTIONS
# --------------------------------------------------

def safe_name(name):
    """Make a string safe to use inside a file name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.lower()).strip("_") or "molecule"


@st.cache_data(ttl=86400, show_spinner="Searching PubChem...")
def lookup_pubchem(name):
    """Find a molecule by name in PubChem. Returns None if not found.

    Network errors are raised, so they are not cached.
    """
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name, safe='')}/property/"
        "IsomericSMILES,IUPACName,MolecularFormula/JSON"
    )
    response = requests.get(url, timeout=10)

    if response.status_code == 404:
        return None

    response.raise_for_status()
    props = response.json()["PropertyTable"]["Properties"][0]

    smiles = (
        props.get("SMILES")
        or props.get("IsomericSMILES")
        or props.get("ConnectivitySMILES")
        or props.get("CanonicalSMILES")
    )

    if not smiles:
        return None

    return {
        "smiles": smiles,
        "iupac": props.get("IUPACName", ""),
        "formula": props.get("MolecularFormula", ""),
    }


def select_molecule(label, key, default_index=0):
    """Let the user pick a built-in molecule, search by name, or enter SMILES.

    Returns a dict with name, smiles, description and category,
    or None if no valid molecule is selected yet.
    """
    source = st.radio(
        f"{label}: choose how to select",
        ["Built-in", "Search by name", "Enter SMILES"],
        horizontal=True,
        key=f"{key}_source",
    )

    if source == "Built-in":
        name = st.selectbox(
            label,
            list(MOLECULES.keys()),
            index=default_index,
            key=f"{key}_builtin",
        )
        data = MOLECULES[name]
        return {
            "name": name,
            "smiles": data["smiles"],
            "description": data["description"],
            "category": data["category"],
        }

    if source == "Search by name":
        query = st.text_input(
            "Molecule name",
            placeholder="e.g. aspirin, caffeine, glucose",
            key=f"{key}_query",
        ).strip()

        if not query:
            st.info("Type a molecule name and press Enter.")
            return None

        try:
            found = lookup_pubchem(query)
        except Exception:
            st.error(
                "Could not reach PubChem. Check your connection and try "
                "again, or use 'Enter SMILES'."
            )
            return None

        if found is None:
            st.warning(
                f"No molecule named '{query}' was found. Check the "
                "spelling or use 'Enter SMILES'."
            )
            return None

        if Chem.MolFromSmiles(found["smiles"]) is None:
            st.error("RDKit could not read the structure returned by PubChem.")
            return None

        description = "Molecule retrieved from the PubChem database."
        if found["iupac"]:
            description += f" IUPAC name: {found['iupac']}."

        return {
            "name": query[:1].upper() + query[1:],
            "smiles": found["smiles"],
            "description": description,
            "category": "PubChem search",
        }

    text = st.text_input(
        "SMILES notation",
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
        key=f"{key}_smiles",
    ).strip()

    if not text:
        st.info("Type a SMILES string and press Enter.")
        return None

    if Chem.MolFromSmiles(text) is None:
        st.error("Invalid SMILES notation. Please check your input.")
        return None

    return {
        "name": "Custom molecule",
        "smiles": text,
        "description": "Molecule entered by the user as a SMILES string.",
        "category": "Custom SMILES",
    }


@st.cache_data
def smiles_to_molblock(smiles):
    """Build a 3D structure (with hydrogens) and return it as a MolBlock."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None or mol.GetNumHeavyAtoms() > 100:
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
    """Show an interactive, rotatable 3D model of a molecule."""
    style_name = st.selectbox(
        "3D style",
        ["Ball and stick", "Stick", "Sphere"],
        key=key,
    )

    styles = {
        "Ball and stick": {
            "stick": {"radius": 0.15},
            "sphere": {"scale": 0.3},
        },
        "Stick": {"stick": {}},
        "Sphere": {"sphere": {}},
    }

    molblock = smiles_to_molblock(smiles)

    if molblock is None:
        st.error(
            "Could not generate a 3D structure for this molecule. "
            "Very large molecules (over 100 heavy atoms) are not supported."
        )
        return

    view = py3Dmol.view(width=500, height=400)
    view.addModel(molblock, "mol")
    view.setStyle({}, styles[style_name])
    view.setBackgroundColor("white")
    view.zoomTo()

    components.html(view._make_html(), height=420)
    st.caption("Drag to rotate. Pinch or scroll to zoom.")


@st.cache_data
def get_properties(name, smiles):
    """Calculate molecular descriptors using RDKit."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "Molecule": name,
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def create_pdf(info):
    """Generate a downloadable PDF report for any molecule."""
    name = info["name"]
    props = get_properties(name, info["smiles"])

    if props is None:
        return None

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph("MolecuLab - Molecular Properties Report", styles["Title"])
    )
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(f"<b>Molecule:</b> {escape(name)}", styles["Normal"])
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph(escape(info["description"]), styles["BodyText"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Molecular Properties", styles["Heading2"]))
    story.append(Spacer(1, 8))

    table_data = [["Property", "Calculated Value"]]

    for key, value in props.items():
        if key != "Molecule":
            table_data.append([key, str(value)])

    table = Table(table_data, colWidths=[240, 220])

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A8B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE5F0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F8FC")),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 16))

    mol = Chem.MolFromSmiles(info["smiles"])

    story.append(Paragraph("SMILES Representation", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(info["smiles"]), styles["Code"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Canonical SMILES: " + escape(Chem.MolToSmiles(mol)),
            styles["Code"],
        )
    )
    story.append(Spacer(1, 16))

    story.append(
        Paragraph(
            "This report was generated using MolecuLab and RDKit. "
            "Calculated descriptors are computational estimates and "
            "do not replace experimental measurements.",
            styles["Italic"],
        )
    )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("# MolecuLab")
st.sidebar.caption("Interactive Cheminformatics Laboratory")
st.sidebar.divider()

st.sidebar.markdown("### Navigation")

page = st.sidebar.selectbox(
    "Choose a section",
    [
        "Home",
        "Theory",
        "Molecular Explorer",
        "Properties Dashboard",
        "Compare Molecules",
        "Chemistry Quiz",
        "Individual PDF Reports",
    ],
    key="main_navigation",
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")
st.sidebar.write("**Domain:** Cheminformatics")
st.sidebar.write("**Toolkit:** RDKit")
st.sidebar.write("**Interface:** Streamlit")

st.sidebar.caption(
    "An educational project for exploring molecular structures "
    "and chemical properties."
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>MolecuLab</h1>
            <p>
                Interactive Cheminformatics Virtual Laboratory
            </p>
            <p>
                Explore molecular structures, calculate chemical
                descriptors, and compare important molecular properties.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Welcome to MolecuLab")

    st.write(
        "MolecuLab is an educational web application that connects "
        "chemistry with computational tools. It helps students "
        "understand how molecular structures can be represented "
        "and analyzed digitally."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Molecular Explorer</h3>
                <p>
                    View 2D and interactive 3D structures of any molecule.
                    Search by name or enter your own SMILES.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Property Analysis</h3>
                <p>
                    Calculate molecular weight, LogP, TPSA,
                    and hydrogen-bond counts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Molecule Comparison</h3>
                <p>
                    Compare descriptors of different molecules
                    using tables and charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Built-in Example Molecules")

    cols = st.columns(3)

    for col, name in zip(cols, MOLECULES.keys()):
        with col:
            st.markdown(f"### {name}")
            st.write(MOLECULES[name]["category"])
            st.write(MOLECULES[name]["description"])

    st.info(
        "Use the Navigation menu in the sidebar to explore "
        "the different laboratory sections."
    )


# --------------------------------------------------
# THEORY PAGE
# --------------------------------------------------

elif page == "Theory":

    st.title("Cheminformatics Theory")

    st.write(
        "Cheminformatics combines chemistry, computer science, "
        "and data analysis to store, represent, and study chemical "
        "structures and their properties."
    )

    with st.expander("1. What is a SMILES notation?"):
        st.write(
            "SMILES stands for Simplified Molecular Input Line Entry "
            "System. It represents a molecular structure as a text string. "
            "This makes it possible to store and process chemical "
            "structures using computers."
        )
        st.code("CCO", language=None)
        st.caption("Example: CCO represents ethanol.")

    with st.expander("2. What is molecular weight?"):
        st.write(
            "Molecular weight is the sum of the atomic weights of "
            "all atoms present in a molecule. It is commonly expressed "
            "in grams per mole (g/mol)."
        )

    with st.expander("3. What is LogP?"):
        st.write(
            "LogP estimates how a neutral compound distributes between "
            "octanol and water. It is often used to discuss relative "
            "lipophilicity. It is not, by itself, a complete measure "
            "of absorption or drug effectiveness."
        )

    with st.expander("4. What is TPSA?"):
        st.write(
            "Topological Polar Surface Area estimates the contribution "
            "of polar atoms to a molecule's surface. It can help "
            "researchers compare molecular properties."
        )

    with st.expander("5. What are hydrogen-bond donors and acceptors?"):
        st.write(
            "Hydrogen-bond donors are groups that can donate a hydrogen "
            "in hydrogen bonding. Acceptors are atoms or groups that "
            "can accept a hydrogen bond."
        )

    with st.expander("6. What is RDKit?"):
        st.write(
            "RDKit is an open-source cheminformatics toolkit. It can "
            "read molecular representations, generate structures, "
            "and calculate many chemical descriptors."
        )

    with st.expander("7. What is a 3D molecular structure?"):
        st.write(
            "A 2D drawing shows which atoms are connected. A 3D "
            "structure also shows how the atoms are arranged in space, "
            "including bond angles and the overall shape of the molecule. "
            "MolecuLab generates 3D coordinates with RDKit, adds "
            "hydrogen atoms, and optimizes the geometry with a force "
            "field. The result is one low-energy conformer, not the only "
            "shape the molecule can adopt."
        )


# --------------------------------------------------
# MOLECULAR EXPLORER
# --------------------------------------------------

elif page == "Molecular Explorer":

    st.title("Molecular Explorer")

    st.write(
        "Choose a built-in molecule, search any molecule by name, "
        "or enter your own SMILES."
    )

    info = select_molecule("Molecule", key="explorer")

    if info is not None:

        mol = Chem.MolFromSmiles(info["smiles"])

        col1, col2 = st.columns([1, 1])

        with col1:
            view_mode = st.radio(
                "View mode",
                ["2D", "3D"],
                horizontal=True,
            )

            if view_mode == "2D":
                st.subheader("2D Molecular Structure")

                image = Draw.MolToImage(mol, size=(500, 400))
                st.image(image, width="stretch")

            else:
                st.subheader("3D Molecular Structure")
                show_3d(info["smiles"], key="style_explorer")

        with col2:
            st.subheader("Molecule Information")

            st.write("**Name:**", info["name"])
            st.write("**Category:**", info["category"])
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))

            st.write("**Description:**")
            st.write(info["description"])

            st.write("**SMILES Representation:**")
            st.code(info["smiles"])

            st.write("**Canonical SMILES (RDKit):**")
            st.code(Chem.MolToSmiles(mol))

            st.download_button(
                "Download SMILES",
                data=info["smiles"],
                file_name=f"{safe_name(info['name'])}_smiles.txt",
                mime="text/plain",
            )


# --------------------------------------------------
# PROPERTIES DASHBOARD
# --------------------------------------------------

elif page == "Properties Dashboard":

    st.title("Molecular Properties Dashboard")

    info = select_molecule("Molecule", key="dashboard")

    if info is not None:

        props = get_properties(info["name"], info["smiles"])

        st.subheader(info["name"])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Molecular Weight",
            f"{props['Molecular Weight (g/mol)']} g/mol",
        )
        col2.metric("LogP", props["LogP"])
        col3.metric("TPSA", f"{props['TPSA (Å²)']} Å²")

        col4, col5, col6 = st.columns(3)

        col4.metric("H-Bond Donors", props["H-Bond Donors"])
        col5.metric("H-Bond Acceptors", props["H-Bond Acceptors"])
        col6.metric("Rotatable Bonds", props["Rotatable Bonds"])

        st.subheader("Complete Property Table")

        property_df = pd.DataFrame(
            {
                "Property": list(props.keys()),
                "Value": [str(v) for v in props.values()],
            }
        )

        st.dataframe(property_df, width="stretch", hide_index=True)

        st.caption(
            "Values are calculated using RDKit. Different software "
            "versions or descriptor definitions may produce "
            "slightly different results."
        )


# --------------------------------------------------
# COMPARE MOLECULES
# --------------------------------------------------

elif page == "Compare Molecules":

    st.title("Compare Molecules")

    st.write(
        "Select two molecules (built-in, searched by name, or entered "
        "as SMILES) to compare their calculated descriptors."
    )

    col1, col2 = st.columns(2)

    with col1:
        info_a = select_molecule("First molecule", key="cmp_a", default_index=0)

    with col2:
        info_b = select_molecule("Second molecule", key="cmp_b", default_index=1)

    if info_a is not None and info_b is not None:

        props_a = get_properties(info_a["name"], info_a["smiles"])
        props_b = get_properties(info_b["name"], info_b["smiles"])

        label_a = info_a["name"]
        label_b = info_b["name"]

        if label_a == label_b:
            label_a, label_b = f"{label_a} (1)", f"{label_b} (2)"

        comparison = pd.DataFrame([props_a, props_b])
        comparison["Molecule"] = [label_a, label_b]

        st.subheader("Comparison Table")

        st.dataframe(comparison, width="stretch", hide_index=True)

        st.subheader("Molecular Weight Comparison")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            [label_a, label_b],
            [
                props_a["Molecular Weight (g/mol)"],
                props_b["Molecular Weight (g/mol)"],
            ],
        )

        ax.set_ylabel("Molecular Weight (g/mol)")
        ax.set_title("Molecular Weight Comparison")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("LogP Comparison")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar([label_a, label_b], [props_a["LogP"], props_b["LogP"]])

        ax.set_ylabel("LogP")
        ax.set_title("Calculated LogP Comparison")

        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download Comparison CSV",
            data=comparison.to_csv(index=False).encode("utf-8-sig"),
            file_name="moleculab_comparison.csv",
            mime="text/csv",
        )

        if info_a["smiles"] == info_b["smiles"]:
            st.info(
                "Both selections refer to the same molecule. "
                "Choose different molecules for a more useful comparison."
            )


# --------------------------------------------------
# CHEMISTRY QUIZ
# --------------------------------------------------

elif page == "Chemistry Quiz":

    st.title("Chemistry Quiz")

    st.write(
        "Test your understanding of molecular representations "
        "and cheminformatics concepts."
    )

    questions = [
        {
            "question": "What does SMILES represent?",
            "options": [
                "A molecular structure as text",
                "A type of microscope",
                "A laboratory temperature",
                "A chemical reaction speed",
            ],
            "answer": "A molecular structure as text",
            "explanation": (
                "SMILES is a text-based representation of molecular structures."
            ),
        },
        {
            "question": "Which toolkit is used in MolecuLab?",
            "options": [
                "TensorFlow",
                "RDKit",
                "OpenCV",
                "Pygame",
            ],
            "answer": "RDKit",
            "explanation": (
                "RDKit provides cheminformatics tools for molecular structures "
                "and descriptor calculations."
            ),
        },
        {
            "question": "Which property is expressed in g/mol?",
            "options": [
                "LogP",
                "TPSA",
                "Molecular weight",
                "Number of atoms",
            ],
            "answer": "Molecular weight",
            "explanation": (
                "Molecular weight is commonly reported in grams per mole."
            ),
        },
        {
            "question": "What does TPSA stand for?",
            "options": [
                "Total Protein Structure Analysis",
                "Topological Polar Surface Area",
                "Thermal Pressure System Analysis",
                "Total Pharmaceutical Safety Assessment",
            ],
            "answer": "Topological Polar Surface Area",
            "explanation": (
                "TPSA estimates the polar surface area associated with "
                "a molecular structure."
            ),
        },
        {
            "question": "Which Python toolkit generates 2D molecular drawings here?",
            "options": [
                "RDKit",
                "ReportLab",
                "Pandas",
                "Streamlit",
            ],
            "answer": "RDKit",
            "explanation": (
                "RDKit can generate 2D depictions of molecular structures."
            ),
        },
    ]

    with st.form("chemistry_quiz"):

        answers = []

        for i, q in enumerate(questions):
            st.markdown(f"### Question {i + 1}")
            st.write(q["question"])

            selected = st.radio(
                "Choose your answer:",
                q["options"],
                key=f"quiz_{i}",
            )

            answers.append(selected)

        submitted = st.form_submit_button("Submit Answers")

    if submitted:

        score = 0

        st.subheader("Quiz Results")

        for i, q in enumerate(questions):

            if answers[i] == q["answer"]:
                score += 1
                st.success(f"Question {i + 1}: Correct!")
            else:
                st.error(
                    f"Question {i + 1}: Correct answer: {q['answer']}"
                )

            st.caption(q["explanation"])

        st.metric(
            "Your Score",
            f"{score} / {len(questions)}",
        )

        if score == len(questions):
            st.balloons()
            st.success("Excellent! You answered every question correctly.")
        elif score >= 3:
            st.info("Good work! Review the explanations to improve further.")
        else:
            st.info("Keep learning! Visit the Theory section and try again.")


# --------------------------------------------------
# INDIVIDUAL PDF REPORTS
# --------------------------------------------------

elif page == "Individual PDF Reports":

    st.title("Individual Molecular Reports")

    st.write(
        "Generate a PDF report containing calculated descriptors "
        "and the SMILES representation of any molecule."
    )

    info = select_molecule("Molecule for the report", key="pdf")

    if info is not None:

        props = get_properties(info["name"], info["smiles"])

        st.subheader("Report Preview")

        st.write("**Molecule:**", info["name"])
        st.write("**Formula:**", props["Molecular Formula"])
        st.write("**Category:**", info["category"])

        st.dataframe(
            pd.DataFrame(
                {
                    "Property": list(props.keys()),
                    "Value": [str(v) for v in props.values()],
                }
            ),
            width="stretch",
            hide_index=True,
        )

        pdf_data = create_pdf(info)

        if pdf_data:

            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"{safe_name(info['name'])}_molecular_report.pdf",
                mime="application/pdf",
            )

    st.caption(
        "Reports are generated for educational purposes. "
        "Computed properties do not establish medical safety, "
        "clinical effectiveness, or biological activity."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "MolecuLab | Interactive Cheminformatics Virtual Laboratory"
)
