import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import AllChem
from io import BytesIO
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MolecuLab | Virtual Chemistry Laboratory",
    page_icon="🧪",
    layout="wide"
)

# --------------------------------------------------
# MOLECULE DATABASE
# --------------------------------------------------

MOLECULES = {
    "Ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "description": (
            "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) "
            "used to relieve pain, fever, and inflammation."
        ),
        "applications": (
            "It is used for temporary relief of pain and inflammation, "
            "including headaches, muscle pain, and some joint conditions."
        ),
        "category": "Pharmaceutical compound",
        "color": "#2563EB"
    },
    "Nicotine": {
        "smiles": "CN1CCC[C@H]1c2cccnc2",
        "description": (
            "Nicotine is an alkaloid naturally found in tobacco plants. "
            "It acts on nicotinic acetylcholine receptors in the nervous system."
        ),
        "applications": (
            "Nicotine is studied in neuroscience and pharmacology. "
            "It is addictive and can cause harmful health effects."
        ),
        "category": "Natural alkaloid",
        "color": "#7C3AED"
    },
    "Quercetin": {
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "description": (
            "Quercetin is a plant flavonoid found in foods such as onions, "
            "apples, and several other fruits and vegetables."
        ),
        "applications": (
            "It is studied in nutritional science and laboratory research "
            "on plant-derived compounds."
        ),
        "category": "Plant flavonoid",
        "color": "#059669"
    }
}

# --------------------------------------------------
# CHEMISTRY FUNCTIONS
# --------------------------------------------------

@st.cache_data
def molecule_data(name):
    smiles = MOLECULES[name]["smiles"]
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid molecular structure for {name}")

    canonical = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {
        "Molecule": name,
        "Molecular Formula": formula,
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Total Atoms": mol.GetNumAtoms(),
        "Bonds": mol.GetNumBonds()
    }

    return mol, canonical, properties


def molecule_image(mol):
    """Create a 2D molecular structure image."""
    copy = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(copy)
    return Draw.MolToImage(copy, size=(650, 420))


def make_pdf(name, student_name, registration):
    """Create a separate PDF report for the selected molecule."""

    mol, canonical, props = molecule_data(name)
    info = MOLECULES[name]

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=25,
        leading=31,
        textColor=colors.HexColor("#174EA6"),
        alignment=TA_CENTER,
        spaceAfter=14
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=19,
        textColor=colors.HexColor("#174EA6"),
        spaceBefore=13,
        spaceAfter=7
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=7
    ))

    story = []

    # Cover heading
    story.append(Paragraph("MolecuLab", styles["ReportTitle"]))
    story.append(Paragraph(
        "INDIVIDUAL MOLECULAR ANALYSIS REPORT",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Molecule:</b> {name}",
        styles["ReportBody"]
    ))
    story.append(Paragraph(
        f"<b>Report date:</b> {date.today().strftime('%B %d, %Y')}",
        styles["ReportBody"]
    ))
    story.append(Paragraph(
        f"<b>Student:</b> {student_name or 'Not provided'}",
        styles["ReportBody"]
    ))
    story.append(Paragraph(
        f"<b>Registration number:</b> "
        f"{registration or 'Not provided'}",
        styles["ReportBody"]
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("1. Introduction", styles["SectionHeading"]))
    story.append(Paragraph(info["description"], styles["ReportBody"]))
    story.append(Paragraph(
        f"<b>Compound category:</b> {info['category']}",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        "2. Molecular Structure", styles["SectionHeading"]
    ))

    image_buffer = BytesIO()
    molecule_image(mol).save(image_buffer, format="PNG")
    image_buffer.seek(0)

    story.append(Image(
        image_buffer,
        width=14 * cm,
        height=9 * cm,
        kind="proportional"
    ))

    story.append(Paragraph(
        "<b>Input SMILES:</b> " + info["smiles"],
        styles["ReportBody"]
    ))
    story.append(Paragraph(
        "<b>Canonical SMILES:</b> " + canonical,
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        "3. Molecular Properties", styles["SectionHeading"]
    ))

    table_data = [["Property", "Calculated value"]]

    for key, value in props.items():
        if key != "Molecule":
            table_data.append([key, str(value)])

    table = Table(
        table_data,
        colWidths=[8 * cm, 7 * cm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174EA6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EFF6FF")]),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(table)

    story.append(Paragraph(
        "4. Chemical Analysis", styles["SectionHeading"]
    ))

    analysis = (
        f"The molecular formula of {name} is {props['Molecular Formula']}. "
        f"Its molecular weight is "
        f"{props['Molecular Weight (g/mol)']} g/mol. "
        f"Its calculated LogP is {props['LogP']}, and its TPSA is "
        f"{props['TPSA (Å²)']} Å². "
        f"The structure contains {props['Heavy Atoms']} heavy atoms and "
        f"{props['Bonds']} bonds. These descriptors help characterize "
        "the molecule but do not independently establish its biological "
        "activity, safety, or effectiveness."
    )

    story.append(Paragraph(analysis, styles["ReportBody"]))

    story.append(Paragraph(
        "5. Applications and Significance", styles["SectionHeading"]
    ))
    story.append(Paragraph(info["applications"], styles["ReportBody"]))

    story.append(Paragraph(
        "6. Conclusion", styles["SectionHeading"]
    ))
    story.append(Paragraph(
        f"This report summarizes the structure and selected "
        f"physicochemical descriptors of {name}. Cheminformatics tools "
        "make it easier to represent, visualize, and compare chemical "
        "molecules using computational methods.",
        styles["ReportBody"]
    ))

    story.append(Paragraph(
        "7. References", styles["SectionHeading"]
    ))
    story.append(Paragraph(
        "1. RDKit documentation: https://www.rdkit.org/docs/<br/>"
        "2. PubChem, National Library of Medicine: "
        "https://pubchem.ncbi.nlm.nih.gov/<br/>"
        "3. Streamlit documentation: https://docs.streamlit.io/",
        styles["ReportBody"]
    ))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(1.8 * cm, 1.0 * cm, "MolecuLab | Educational Report")
        canvas.drawRightString(
            A4[0] - 1.8 * cm,
            1.0 * cm,
            f"Page {document.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --------------------------------------------------
# PROFESSIONAL WHITE AND BLUE DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
        color: #172B4D;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #EAF2FF, #FFFFFF);
        border: 1px solid #D7E5FF;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #174EA6;
        font-size: 2.4rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #475569;
        font-size: 1.05rem;
    }

    .molecule-card {
        background: #FFFFFF;
        padding: 20px;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        min-height: 180px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .molecule-card h3 {
        color: #174EA6;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 12px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid #2563EB;
    }

    h1, h2, h3 {
        color: #174EA6;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("# MolecuLab")
st.sidebar.caption("Interactive Cheminformatics Laboratory")
st.sidebar.divider()

page = st.sidebar.radio(
    "EXPLORE",
    [
        "Home",
        "Theory",
        "Molecular Explorer",
        "Properties Dashboard",
        "Compare Molecules",
        "Chemistry Quiz",
        "Individual PDF Reports"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Student Information")

student_name = st.sidebar.text_input(
    "Student name",
    key="student_name"
)

registration = st.sidebar.text_input(
    "Registration number",
    key="registration"
)

st.sidebar.caption("Computational chemistry learning tool")


# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>MolecuLab</h1>
        <p>
        Explore molecular structures, discover chemical properties,
        and understand the science behind important molecules.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Your Molecular Workspace")

    cols = st.columns(3)

    for col, name in zip(cols, MOLECULES):
        info = MOLECULES[name]

        with col:
            st.markdown(
                f"""
                <div class="molecule-card">
                    <h3>{name}</h3>
                    <p>{info['category']}</p>
                    <p>{info['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.subheader("Explore the Laboratory")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**01 — Molecular Explorer**")
        st.write("Inspect structures and SMILES notation.")

    with c2:
        st.markdown("**02 — Properties Dashboard**")
        st.write("Explore important molecular descriptors.")

    with c3:
        st.markdown("**03 — Individual Reports**")
        st.write("Generate a separate PDF for each molecule.")

    st.info(
        "Choose a section from the left navigation menu to begin."
    )


# --------------------------------------------------
# THEORY
# --------------------------------------------------

elif page == "Theory":

    st.title("Chemistry Foundations")
    st.write(
        "Cheminformatics combines chemistry, computation, and data analysis "
        "to represent and study chemical structures."
    )

    with st.expander("What is SMILES?"):
        st.write(
            "SMILES stands for Simplified Molecular Input Line Entry System. "
            "It represents a chemical structure using a string of characters."
        )

    with st.expander("What is molecular weight?"):
        st.write(
            "Molecular weight is the sum of the atomic weights of the atoms "
            "in a molecule. It is commonly expressed in g/mol."
        )

    with st.expander("What is LogP?"):
        st.write(
            "LogP estimates how a neutral molecule partitions between "
            "octanol and water. It is commonly used as a descriptor of "
            "lipophilicity."
        )

    with st.expander("What is TPSA?"):
        st.write(
            "Topological polar surface area estimates the contribution of "
            "polar atoms to a molecule's surface. It can help characterize "
            "molecular polarity."
        )

    with st.expander("What are hydrogen bond donors and acceptors?"):
        st.write(
            "These descriptors estimate the groups that can donate or accept "
            "hydrogen bonds, which can influence molecular interactions."
        )


# --------------------------------------------------
# MOLECULAR EXPLORER
# --------------------------------------------------

elif page == "Molecular Explorer":

    st.title("Molecular Explorer")

    name = st.selectbox("Choose a molecule", list(MOLECULES))
    mol, canonical, props = molecule_data(name)

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("2D Molecular Structure")
        st.image(molecule_image(mol), use_container_width=True)

    with right:
        st.subheader(name)
        st.write(MOLECULES[name]["description"])
        st.caption(f"Category: {MOLECULES[name]['category']}")

        st.markdown("**Molecular formula**")
        st.code(props["Molecular Formula"])

        st.markdown("**Input SMILES**")
        st.code(MOLECULES[name]["smiles"])

        st.markdown("**Canonical SMILES**")
        st.code(canonical)

    st.subheader("Structure Information")

    a, b, c = st.columns(3)
    a.metric("Heavy atoms", props["Heavy Atoms"])
    b.metric("Total atoms", props["Total Atoms"])
    c.metric("Bonds", props["Bonds"])

    st.subheader("Validate Another SMILES")

    custom_smiles = st.text_input(
        "Enter a SMILES string",
        placeholder="Enter SMILES notation here"
    )

    if st.button("Validate structure"):

        if not custom_smiles.strip():
            st.warning("Enter a SMILES string first.")
        else:
            custom_mol = Chem.MolFromSmiles(custom_smiles)

            if custom_mol is None:
                st.error("The supplied SMILES could not be parsed.")
            else:
                st.success("Valid SMILES structure.")
                st.image(
                    molecule_image(custom_mol),
                    caption="Your molecular structure",
                    use_container_width=True
                )
                st.write(
                    "Formula:",
                    rdMolDescriptors.CalcMolFormula(custom_mol)
                )
                st.write(
                    "Canonical SMILES:",
                    Chem.MolToSmiles(custom_mol)
                )


# --------------------------------------------------
# PROPERTIES DASHBOARD
# --------------------------------------------------

elif page == "Properties Dashboard":

    st.title("Molecular Properties Dashboard")

    name = st.selectbox("Select a molecule", list(MOLECULES))
    mol, canonical, props = molecule_data(name)

    st.subheader(name)

    c1, c2, c3 = st.columns(3)

    c1.metric("Molecular Weight", f"{props['Molecular Weight (g/mol)']} g/mol")
    c2.metric("LogP", props["LogP"])
    c3.metric("TPSA", f"{props['TPSA (Å²)']} Å²")

    c4, c5, c6 = st.columns(3)

    c4.metric("H-Bond Donors", props["H-Bond Donors"])
    c5.metric("H-Bond Acceptors", props["H-Bond Acceptors"])
    c6.metric("Heavy Atoms", props["Heavy Atoms"])

    st.info(
        "These values are computed using RDKit. They are molecular "
        "descriptors, not direct measurements of clinical effectiveness "
        "or safety."
    )

    st.subheader("Complete Property Table")

    single_df = pd.DataFrame([props])
    st.dataframe(single_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download this molecule's properties (CSV)",
        single_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{name.lower()}_properties.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# COMPARE MOLECULES
# --------------------------------------------------

elif page == "Compare Molecules":

    st.title("Compare Molecules")

    left_name = st.selectbox(
        "First molecule",
        list(MOLECULES),
        index=0
    )

    right_name = st.selectbox(
        "Second molecule",
        list(MOLECULES),
        index=1
    )

    left_props = molecule_data(left_name)[2]
    right_props = molecule_data(right_name)[2]

    comparison = pd.DataFrame({
        left_name: left_props,
        right_name: right_props
    })

    st.subheader("Side-by-Side Comparison")

    st.dataframe(comparison, use_container_width=True)

    numeric_properties = [
        "Molecular Weight (g/mol)",
        "LogP",
        "TPSA (Å²)",
        "H-Bond Donors",
        "H-Bond Acceptors",
        "Heavy Atoms",
        "Bonds"
    ]

    selected_property = st.selectbox(
        "Property to visualize",
        numeric_properties
    )

    chart_df = pd.DataFrame({
        "Molecule": [left_name, right_name],
        "Value": [
            left_props[selected_property],
            right_props[selected_property]
        ]
    })

    st.bar_chart(chart_df.set_index("Molecule"))

    difference = (
        right_props[selected_property] -
        left_props[selected_property]
    )

    st.write(
        f"The difference in {selected_property} "
        f"({right_name} minus {left_name}) is "
        f"{difference:.2f}."
    )


# --------------------------------------------------
# CHEMISTRY QUIZ
# --------------------------------------------------

elif page == "Chemistry Quiz":

    st.title("Chemistry Knowledge Check")

    questions = [
        (
            "What does SMILES represent?",
            [
                "A text representation of molecular structure",
                "A microscope",
                "A biological cell"
            ],
            0,
            "SMILES represents molecular structures as text."
        ),
        (
            "Which property represents molecular mass?",
            [
                "TPSA",
                "Molecular weight",
                "LogP"
            ],
            1,
            "Molecular weight is commonly expressed in g/mol."
        ),
        (
            "What does TPSA describe?",
            [
                "Topological polar surface area",
                "Molecular color",
                "Number of chemical samples"
            ],
            0,
            "TPSA is a descriptor of molecular polarity."
        ),
        (
            "Which of these is a plant flavonoid?",
            [
                "Ibuprofen",
                "Nicotine",
                "Quercetin"
            ],
            2,
            "Quercetin is a plant-derived flavonoid."
        ),
        (
            "What does LogP help estimate?",
            [
                "Molecular age",
                "Partitioning between octanol and water",
                "Number of chemical bonds"
            ],
            1,
            "LogP is commonly used to characterize lipophilicity."
        )
    ]

    with st.form("moleculab_quiz"):

        answers = []

        for i, (question, options, correct, explanation) in enumerate(questions):
            answer = st.radio(
                f"{i + 1}. {question}",
                options,
                index=None,
                key=f"quiz_{i}"
            )
            answers.append(answer)

        submitted = st.form_submit_button("Check My Answers")

    if submitted:

        score = sum(
            answer == options[correct]
            for answer, (question, options, correct, explanation)
            in zip(answers, questions)
        )

        st.success(f"Your score: {score}/{len(questions)}")

        for i, (question, options, correct, explanation) in enumerate(questions):
            if answers[i] == options[correct]:
                st.markdown(f"**Question {i + 1}: Correct**")
            else:
                st.markdown(
                    f"**Question {i + 1}: Correct answer — "
                    f"{options[correct]}**"
                )

            st.write(explanation)


# --------------------------------------------------
# INDIVIDUAL PDF REPORTS
# --------------------------------------------------

elif page == "Individual PDF Reports":

    st.title("Individual Molecular Report Generator")

    st.write(
        "Generate a separate scientific report for one selected molecule. "
        "Each downloaded PDF contains information about that molecule only."
    )

    name = st.selectbox(
        "Select the molecule for your report",
        list(MOLECULES),
        key="report_molecule"
    )

    mol, canonical, props = molecule_data(name)

    st.subheader(f"{name} — Report Preview")

    left, right = st.columns([1, 1])

    with left:
        st.image(
            molecule_image(mol),
            caption=f"2D structure of {name}",
            use_container_width=True
        )

    with right:
        st.markdown(f"**Formula:** {props['Molecular Formula']}")
        st.markdown(
            f"**Molecular weight:** "
            f"{props['Molecular Weight (g/mol)']} g/mol"
        )
        st.markdown(f"**LogP:** {props['LogP']}")
        st.markdown(f"**TPSA:** {props['TPSA (Å²)']} Å²")

        st.write(MOLECULES[name]["description"])

    st.divider()

    st.subheader("Download Your Individual PDF")

    st.write(
        f"Your report will include the introduction, molecular structure, "
        f"SMILES notation, calculated properties, chemical analysis, "
        f"applications, conclusion, and references for {name}."
    )

    if st.button("Generate PDF Report", type="primary"):

        try:
            pdf_data = make_pdf(
                name,
                student_name,
                registration
            )

            st.download_button(
                label=f"Download {name} Molecular Report",
                data=pdf_data,
                file_name=f"{name}_Molecular_Report.pdf",
                mime="application/pdf",
                key=f"download_{name}"
            )

            st.success(
                f"Your individual {name} report has been generated."
            )

        except Exception as error:
            st.error(
                "The PDF could not be generated. Please try again."
            )
            st.exception(error)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "MolecuLab | Interactive Cheminformatics Virtual Laboratory"
)