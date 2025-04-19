
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings

st.title("KG Construction App")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
else:
    st.stop()

#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import Document
embeddings = OpenAIEmbeddings()
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from tempfile import NamedTemporaryFile

msgs = ChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

def process_pdf(pdf_path):
    all_results1 = []
    all_results2 = []
    all_results = []

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings)
    store = InMemoryStore()

    big_chunks_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    msgs = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    template1 =""" You are a Materials Science assistant in the field of hydrogen technologies, specializing in extracting entities and relationships for knowledge graphs using the <http://emmo.info/emmo#> ontology. Your expertise lies in the fabrication workflow and materials in Materials Science. Use the provided context to answer the question at the end. If unsure, state that you don't know rather than conjecturing.

Use the unique ID for any newly created entities. Extract the nodes based on these definitions:

- Material: is defined as a substance used to construct or compose objects, possessing specific physical and chemical properties. These properties determine the material's behavior under different conditions and its suitability for various applications. Materials can include chemicals, intermediates, products, components, molecules, atoms, devices, solvents and surfactants." ;
            Materials are both the raw materials used as input for the main manufacturing process and the product materials as output of the manufacturing process." ;
            "surfactants, chemicals, intermediates, products, components, molecules, atoms, devices and solvents are material."

- Manufacturing: is the process of converting raw materials into finished products using various physical, chemical and mechanical processes. In the context of hydrogen technologies, manufacturing includes the production of components such as fuel cells, electrolysers and hydrogen storage systems as well as the assembly of these components into complete systems." ;
                It is considered the most important manufacturing process." ;
                "HydrogenComponentManufacturing", "ManufacturingProcesses", "Assembly", "Fabrication", "Production"


- Measurement: is the process of experimentally obtaining one or more measurement results that can reasonably be attributed to a quantity.Measurement is the process of obtaining quantitative or qualitative data about a material or system. In hydrogen technologies, measurements are used to characterize the properties and performance of materials, devices, and systems." ;
             "DataAcquisition", "ExperimentalAnalysis", "AnalyticalMeasurement", "PerformanceTesting", "MaterialCharacterization"



- Property: is a characteristic or attribute that describes the behavior, performance, or nature of a material. These properties can be broadly categorized into physical (e.g., density, color), mechanical (e.g., strength, hardness), thermal (e.g., conductivity, expansion), and chemical (e.g., reactivity, corrosion resistance)."
          property has a numeric value with a unit.

- Parameter: refers to a specific and quantifiable and measurable factor in the manufacturing process"@en ;
           parameter has a numeric value and the unit.
           "each manufacturing process has its own parameter."

**node extraction rules**
Follow this struction for extracting the entities:
- Look for any example of materials entity that should be defined as nodes in the knowledge graph with minimal details necessary for understanding the relationship for the knowledge graph
- Materials including surfactants, chemicals, intermediates, products, components, molecules, atoms, devices and solvents.
- Look for main manufacturing that should be defined as nodes in the knowledge graph with minimal details necessary for understanding the relationship
- Extract the name of main manufacturing and ignore all the processes in between
- Look for measurement process that should be defined as nodes in the knowledge graph with minimal details necessary for understanding the relationship
- If a property has multiple distinct values for a material, each value should be represented by a separate property node in the knowledge graph
- If different materials have the same property, each of them should be represented by a separate node in the knowledge graph
- The value of the property must be numeric and accompanied by a unit, and cannot be a string or text
- Parameter for each of the manufacturing should be represented by a separate node in the knowledge graph
- If a parameter has multiple distinct values, each value should be represented by a separate node in the knowledge graph
- The value of the parameter must be numeric and accompanied by a unit, and cannot be a string or text

ALWAYS REMEMBER:

- Extract all the nodes that should be defined as nodes in the knowledge graph with minimal details necessary for understanding the relationship
- Always remember that nodes are newly created entities that should extract from the text.
- Is really IMPORTANT to extract all the nodes. as you are material science expert you know that all surfactants,solutions, chemicals, intermediates, products, components, molecules, atoms, devices and solvents are material.
- As you are material science expert you know that is important to extract all the parameters related to each manufctring and all the properties related to each materials, and extracting them complete and precise.
- focusing solely on extracting nodes for knowledge graph, such as 'Material', 'Manufacturing', 'Measurement', 'Parameter', and 'Property'.
- always double check to extract all the entities needed for creating relationships in the knowledge graph
- Avoid extra explanations; directly format the output as:

"Material": [] # it can be a list
"Manufacturing": [] # it can be a list
"Measurement": [] # it can be a list
"Property": [] # it can be a list
"Parameter": [] # it can be a list
** Here are some Examples to help you how extract the nodes from a text. you should NEVER use the context of this examples to answer the questions, because these are just examples to help you to better understand the task.

- Example for Extracting Materials from this text in Angle brackets:
<In the production of silicon carbide (SiC) nanowires, a mixture containing silicon dioxide (SiO2), graphite powder, and iron (Fe) as a catalyst was used.
The mixture was placed in a furnace and heated to 1400°C in an argon atmosphere. After the reaction, the product was allowed to cool down to
room temperature. The SiC nanowires were then separated from the rest of the mixture through a series of washing and centrifugation steps.
Finally, the nanowires were dried under vacuum conditions.>
In this example, the raw materials are Silicon Dioxide (SiO2), Graphite Powder, and Iron (Fe), and the output material is silicon carbide (SiC) nanowires as the product of synthesis.
So, in this example, the materials are: 'Silicon Dioxide', 'Graphite Powder', 'Iron', 'Silicon Carbide Nanowires'.

- Example for Extracting Manufacturing from this text in Angle brackets:
<In the fabrication of a high-efficiency fuel cell, a proton exchange membrane (PEM) is synthesized from raw polytetrafluoroethylene (PTFE) as
the base material. The membrane electrode assembly (MEA) is then constructed by coating the PEM with a catalyst layer composed of
platinum on carbon (Pt/C), followed by the assembly of the anode and cathode plates on either side. The entire assembly is then compressed and
heated to ensure proper adhesion and electrical contact. Subsequently, the fuel cell is integrated into a hydrogen storage system, which involves
connecting the fuel cell to hydrogen tanks equipped with advanced pressure regulation systems. This integration process is crucial for optimizing the fuel cell's performance and ensuring safe hydrogen storage and delivery.>

In this example, the manufacturing process involves several key steps aligned with the ontology definitions:
So, in this example, the manufacturing processes are: 'Synthesis', 'Construction', 'Compression and Heating', 'Integration'.

- Example for Extracting Measurementfrom this text in Angle brackets:
<In the evaluation of a novel electrolyzer catalyst, the working electrode was coated with a thin layer of iridium oxide (IrO2) and the counter
electrode was made of platinum (Pt). These electrodes were separated by a proton exchange membrane (PEM). The reference electrode used in this setup
was a standard calomel electrode (SCE). Electrochemical impedance spectroscopy (EIS) measurements were conducted to assess the catalyst's efficiency
and durability. The EIS was performed at an open circuit potential with a frequency range from 10 Hz to 100 kHz and an AC amplitude of 5 mV in a 0.5 M
sulfuric acid (H2SO4) solution under an oxygen atmosphere. The starting frequency was set at 100 kHz, decreasing to 10 Hz to fully capture the impedance
response of the catalyst at various frequencies.>

In this example, the measurement process is Electrochemical Impedance Spectroscopy (EIS). So, in this example measurement process is: 'Electrochemical Impedance Spectroscopy (EIS)'

- Example for Extracting Properties:
<Photoluminescence spectroscopy was performed on quantum dot samples to evaluate their optical properties.
The emission peak observed at 520 nm indicates the presence of CdSe quantum dots, which is consistent with their expected size and composition.
The quantum yield of the sample was calculated to be approximately 50%, demonstrating a high level of luminescence efficiency.
Additionally, the sample showed a narrow emission bandwidth of 25 nm, indicating a uniform size distribution of the quantum dots.>

In this example, the properties are: 'emission peak, value: 520 nm', 'quantum yield, value: 50%', 'emission bandwidth, value: 25 nm'.

- Example for Extracting Parameters:
<In the process of fabricating thin-film transistors, a solution of 2,8-difluoro-5,11- bis(triethylsilylethynyl)anthradithiophene (DTE)
in chlorobenzene was spin-coated onto a substrate at 2000 rpm for 60 seconds. The substrate was then baked at 90°C for 5 minutes to remove any solvent
residues. Subsequently, the film underwent annealing at 150°C for 30 minutes to improve its crystallinity. The thickness of the resulting film was
measured to be approximately 45 nm.>

In this example, the parameters are: 'spin-coating speed, value: 2000 rpm', 'spin-coating time, value: 60 seconds', 'baking temperature, value: 90°C',
'baking time, value: 5 minutes', 'annealing temperature, value: 150°C', 'annealing time, value: 30 minutes', 'film thickness, value: 45 nm'.


{chat_history}
{context}
Question: {question}
."""
    prompt1 = PromptTemplate(
        template=template1, input_variables=["context", "question", "chat_history"], output_parser=CommaSeparatedListOutputParser()
    )

    template2 = """ You are a Materials Science assistant in the field of hydrogen technologies, specializing in extracting entities and relationships within knowledge graphs using the <http://emmo.info/emmo#> ontology. Your expertise lies in understanding the fabrication workflow and materials in Materials Science. Use the provided context to answer the question at the end. If unsure, state that you don't know rather than conjecturing.
Extract the relationships between the nodes based on these definitions:

"is_manufacturing_input": signifies the primary material or raw substance that serves as the starting point for a manufacturing process. This material undergoes various stages of processing, treatment, or manipulation to yield the final product.
    domain :Material;
    range :Manufacturing;

"has_manufacturing_output": refers to the final product or material that is generated as a result of a manufacturing process. This output represents the transformed or processed form of the initial raw or semi-processed material.

    domain :Manufacturing;
    range :Material;

"is_measurement_input": refers to the initial set of parameters or characteristics that are subject to quantitative assessment or measurement. These parameters serve as the basis for collecting data and obtaining specific numerical values during the measurement process.
    domain :Material;
     range :Measurement;

"has_measurement_output": " refers to the quantifiable results or data obtained through a measurement process applied to a material or a set of parameters. These outputs represent the numerical values or characteristics that have been assessed, observed, or calculated during the measurement procedure. The has_measurement_output can include measurements of various material properties such as dimensions, mass, density, hardness, and other relevant attributes, providing valuable information for characterizing and analyzing materials.
    domain :Measurement;
    range :Property;

 "has_property": denotes a specific characteristic, trait, or measurable attribute inherent to a material. Properties in materials science encompass a broad range of features, including but not limited to dimensions, mass, density, hardness, thermal conductivity, electrical conductivity, and other quantifiable attributes. These properties define how a material behaves under various conditions and influence its suitability for specific applications.
    domain :Material;
    range :Property;

"has_parameter": refers to a specific and quantifiable attribute or characteristic in the manufacturing process. Parameters in materials science may include various properties such as dimensions, mass, density, hardness, thermal conductivity, and other measurable features.
    domain :Manufacturing;
    range :Parameter;


ALWAYS REMEMBER:
- Focusing solely on extracting relationships between nodes.
- If new nodes are identified during relationship extraction, include them in the output with minimal details necessary for understanding the relationship.
- Avoid extra explanations; directly format the output as:

"is_manufacturing_input": [] # it can be a list
"has_manufacturing_output": [] # it can be a list
"is_measurement_input": [] # it can be a list
"has_measurement_output": [] # it can be a list
"has_property": [] # it can be a list
"has_parameter": [] # it can be a list

**Relationships extraction rules**
As you contribute to building the knowledge graph, following the established rules is crucial. These rules ensure that the graph accurately represents the domain knowledge, maintains structural integrity, and adheres to the EMMO ontology. Please read and apply the following guidelines carefully when extracting the relationships:

- Every node needs to have at least on edge to another node.
- The edge "is_manufacturing_input" connects material nodes and the manufacturing nodes.
- The edge "is_manufacturing_output" connects manufacturing nodes and the material nodes.
- material nodes cannot have an 'is_manufacturing_input' and 'is_manufacturing_output' edge with the same manufacturing node.
- material nodes cannot have an 'has_manufacturing_output' relationship with two different manufacturing nodes.
- Make sure the relationships are logical and adhere to materials science concepts.
- Each property needs share exactly ONE 'has_property' edge with a material node.
- Pair manufacturing and measurement nodes with parameter nodes with "has_parameter" edges.
- NEVER CONNECT one parameter with more than one manufacturing/measurement nodes.
- Every measurement node needs to have at least on edge another property node.
- Each property node can only share a 'has_measurement_output' with one measurement node.
- The edge "has_measurement_output" connects measurement nodes and property nodes.
- Each triple should follow the format: (subject, predicate, object).
- subject and objects of the triplets are the nodes which extracted and stored in the history.
- Avoid ANY extra explanation of about node and relationship, JUST give the output in the above format, without even a sentence of explanation.
- If asked, extract relationship only that relationship, NOT all the relationships.
- If you have extracted the node and relationship in the format above, you don't need to explain each relationship again separately



{chat_history}
{context}
Question: {question}


."""

    prompt2 = PromptTemplate(
        template=template2, input_variables=["context", "question", "chat_history"], output_parser=CommaSeparatedListOutputParser()
    )

    template3 =""" You are a Materials Science assistant in the field of hydrogen technologies, specializing in extracting nodes and edges for the knowledge graphs using the <http://emmo.info/emmo#> ontology, formatted as json-ld. Your expertise lies in the fabrication workflow and materials in Materials Science. Use the provided context to answer the question at the end. If unsure, state that you don't know rather than conjecturing.

Use the prefix "ex:" with IRI <http://example.com/> for any newly created entities that stored in chat_history with a unique ID. Only utilize classes from the <http://emmo.info/emmo#> ontology as defined.

IMPORTANT: You MUST output the complete knowledge graph, including ALL nodes and relationships. Do not truncate or cut off the output. If the output is large, ensure it is complete and properly formatted.

@prefix : <http://emmo.info/emmo#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix ex: <http://example.com/> .

# Context definition
@context {{
  "emmo": "http://emmo.info/emmo#",
  "is_manufacturing_input": "emmo:EMMO_e1097637",
  "has_manufacturing_output": "emmo:EMMO_e1245987",
  "is_measurement_input": "emmo:EMMO_m5677989",
  "has_measurement_output": "emmo:EMMO_m87987545",
  "has_property": "emmo:EMMO_p5778r78",
  "has_parameter": "emmo:EMMO_p46903ar7",
  "Material": "emmo:EMMO_4207e895_8b83_4318_996a_72cfb32acd94",
  "Manufacturing": "emmo:EMMO_a4d66059_5dd3_4b90_b4cb_10960559441b",
  "Measurement": "emmo:EMMO_463bcfda_867b_41d9_a967_211d4d437cfb",
  "Property": "emmo:EMMO_b7bcff25_ffc3_474e_9ab5_01b1664bd4ba",
  "Parameter": "emmo:EMMO_d1d436e7_72fc_49cd_863b_7bfb4ba5276a"
}}

# Graph data
@graph [
{chat_history}
{context}
]

Question: {question}
"""
    prompt3 = PromptTemplate(
        template=template3, input_variables=["context", "question", "chat_history"], output_parser=CommaSeparatedListOutputParser()
    )
    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        temperature=0,
        max_tokens=4000,
    )

    #print(f"Processing PDF: {pdf_path}")

    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()

    for idx, doc in enumerate(docs):


        headers_to_split_on = [
            ("Abstract", "Header 1"),
            ("Introduction", "Header 2"),
            ("Experimental Methods", "Header 3"),
            ("Results and Discussion", "Header 4"),
            ("Conclusion", "Header 5"),
            ("References", "Header 6")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        big_chunks_retriever.add_documents(md_header_splits)


    # Build QA interface for the PDF
    qa_interface1 = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=big_chunks_retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt1}
    )
     # Build QA interface for the PDF
    qa_interface2 = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=big_chunks_retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt2}
    )
    qa_interface3 = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=big_chunks_retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt3}
    )
    query1 = "extract the materials, manufacturing, measurement, parameters, properties from each document ."
    result = qa_interface1({"question": query1})
    all_results1.append(result["answer"].strip('```json').strip('```').strip())

    queries2 = [
        "extract the relationship between materials and manufacturing as is_manufacturing_input and is_manufacturing_output. Avoid any extra explanation.",
        "extract the relationship between materials and measurement as is_measurement_input and relationship between measurement and property as has_measurement_output. Avoid any extra explanation.",
        "extract the relationship between materials and property as has_property. Avoid any extra explanation.",
        "extract the relationship between manufacturing and parameter as has_parameter. Avoid any extra explanation.",
    ]

    for query in queries2:
        result2 = qa_interface2({"question": query})
        all_results2.append(result2["answer"].strip('```json').strip('```').strip())

    query3 = "Create a complete knowledge graph including all the nodes and the relationships extracted and stored in History. Use the URI defined for 'Classes' and 'Object properties' in emmo ontology. Format the output in JSON-LD."
    result = qa_interface3({"question": query3})
    all_results.append(result["answer"].strip('```json').strip('```').strip())

    #print(f"Finished Processing PDF: {pdf_path}")
    msgs.clear()
    memory.clear()

    return all_results

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    try:
        results = process_pdf(tmp_path)
        for result in results:
            st.write(result)
    finally:
        os.remove(tmp_path)
