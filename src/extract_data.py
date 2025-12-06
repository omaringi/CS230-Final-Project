import os
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_pdf_text(filepath):
    """Extract text from PDF"""
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text[:10000]
    except Exception as e:
        print(f"Error with {filepath}: {e}")
        return ""

def extract_docx_text(filepath):
    """Extract text from Word doc"""
    try:
        doc = Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])[:10000]
    except Exception as e:
        print(f"Error with {filepath}: {e}")
        return ""

def extract_xlsx_text(filepath):
    """Extract text from Excel"""
    try:
        df = pd.read_excel(filepath, nrows=50)
        return " ".join(df.columns.astype(str)) + " " + df.to_string()[:5000]
    except Exception as e:
        print(f"Error with {filepath}: {e}")
        return ""

def get_ctd_section_from_path(filepath):
    """Extract CTD section label from folder path"""
    path_str = str(filepath).lower()
    
    # Module 2 - Clinical/Nonclinical Overviews
    if "2.4" in path_str or "nonclinical overview" in path_str:
        return "2.4"
    elif "2.5" in path_str or "clinical overview" in path_str:
        return "2.5"
    elif "2.6" in path_str or "nonclinical summary" in path_str:
        return "2.6"
    elif "2.7" in path_str or "clinical summary" in path_str:
        return "2.7"
    
    # Module 3 - Quality (Drug Substance)
    elif "3.2.s.1" in path_str:
        return "3.2.S.1"
    elif "3.2.s.2" in path_str:
        return "3.2.S.2"
    elif "3.2.s.3" in path_str:
        return "3.2.S.3"
    elif "3.2.s.4" in path_str:
        return "3.2.S.4"
    elif "3.2.s.5" in path_str:
        return "3.2.S.5"
    elif "3.2.s.6" in path_str:
        return "3.2.S.6"
    elif "3.2.s.7" in path_str:
        return "3.2.S.7"
    
    # Module 3 - Quality (Drug Product)
    elif "3.2.p.1" in path_str:
        return "3.2.P.1"
    elif "3.2.p.2.2" in path_str or "dissolution" in path_str:
        return "3.2.P.2.2"
    elif "3.2.p.2" in path_str or "pharmaceutical development" in path_str:
        return "3.2.P.2"
    elif "3.2.p.3" in path_str or "manufactur" in path_str:
        return "3.2.P.3"
    elif "3.2.p.4" in path_str or "excipient" in path_str:
        return "3.2.P.4"
    elif "3.2.p.5" in path_str or "control of drug product" in path_str:
        return "3.2.P.5"
    elif "3.2.p.6" in path_str or "reference standard" in path_str:
        return "3.2.P.6"
    elif "3.2.p.7" in path_str or "container" in path_str:
        return "3.2.P.7"
    elif "3.2.p.8" in path_str or "stability" in path_str:
        return "3.2.P.8"
    
    # Module 5 - Clinical
    elif "5.3.1" in path_str or "bioavailability" in path_str or "bioequivalence" in path_str:
        return "5.3.1"
    elif "5.3.5" in path_str:
        return "5.3.5"
    elif "5.4" in path_str or "literature" in path_str:
        return "5.4"
    
    return None

def get_section_from_filename(filename):
    """Extract CTD section from final dossier filename"""
    filename_lower = filename.lower()
    
    # Common patterns in final dossier filenames
    patterns = {
        "3.2.p.1": "3.2.P.1",
        "3.2.p.2": "3.2.P.2",
        "3.2.p.3": "3.2.P.3",
        "3.2.p.4": "3.2.P.4",
        "3.2.p.5": "3.2.P.5",
        "3.2.p.6": "3.2.P.6",
        "3.2.p.7": "3.2.P.7",
        "3.2.p.8": "3.2.P.8",
        "3.2.s.1": "3.2.S.1",
        "3.2.s.2": "3.2.S.2",
        "3.2.s.3": "3.2.S.3",
        "3.2.s.4": "3.2.S.4",
        "3.2.s.5": "3.2.S.5",
        "3.2.s.6": "3.2.S.6",
        "3.2.s.7": "3.2.S.7",
        "2.3": "2.3",
        "2.4": "2.4",
        "2.5": "2.5",
        "2.6": "2.6",
        "2.7": "2.7",
    }
    
    for pattern, section in patterns.items():
        if pattern in filename_lower:
            return section
    
    return None

def process_source_documents(base_path):
    """Process source documents (original function)"""
    records = []
    base = Path(base_path)
    
    extensions = ['*.pdf', '*.docx', '*.xlsx', '*.doc']
    all_files = []
    for ext in extensions:
        all_files.extend(base.rglob(ext))
    
    print(f"Found {len(all_files)} source files")
    
    for filepath in tqdm(all_files, desc="Extracting source docs"):
        # Skip final dossier chapters
        if "final module" in str(filepath).lower():
            continue
        if "final sequence" in str(filepath).lower():
            continue
        if "guideline" in str(filepath).lower():
            continue
            
        section = get_ctd_section_from_path(filepath)
        if section is None:
            continue
        
        ext = filepath.suffix.lower()
        if ext == '.pdf':
            text = extract_pdf_text(str(filepath))
        elif ext in ['.docx', '.doc']:
            text = extract_docx_text(str(filepath))
        elif ext == '.xlsx':
            text = extract_xlsx_text(str(filepath))
        else:
            continue
        
        if len(text) < 100:
            continue
            
        records.append({
            'filename': filepath.name,
            'filepath': str(filepath),
            'text': text,
            'section': section,
            'data_type': 'source'
        })
    
    return records

def process_final_dossier(dossier_path):
    """Process final dossier chapters - GOLD STANDARD examples"""
    records = []
    base = Path(dossier_path)
    
    extensions = ['*.pdf', '*.docx', '*.doc']
    all_files = []
    for ext in extensions:
        all_files.extend(base.rglob(ext))
    
    print(f"Found {len(all_files)} final dossier files")
    
    for filepath in tqdm(all_files, desc="Extracting final dossier"):
        # Try to get section from filename first
        section = get_section_from_filename(filepath.name)
        
        # If not in filename, try path
        if section is None:
            section = get_ctd_section_from_path(filepath)
        
        if section is None:
            print(f"  Skipping (no section found): {filepath.name}")
            continue
        
        ext = filepath.suffix.lower()
        if ext == '.pdf':
            text = extract_pdf_text(str(filepath))
        elif ext in ['.docx', '.doc']:
            text = extract_docx_text(str(filepath))
        else:
            continue
        
        if len(text) < 100:
            continue
        
        # Split long documents into chunks for more training examples
        if len(text) > 5000:
            chunks = [text[i:i+5000] for i in range(0, len(text), 4000)]
            for i, chunk in enumerate(chunks[:5]):  # Max 5 chunks per doc
                if len(chunk) > 500:
                    records.append({
                        'filename': f"{filepath.name}_chunk{i}",
                        'filepath': str(filepath),
                        'text': chunk,
                        'section': section,
                        'data_type': 'final_dossier'
                    })
        else:
            records.append({
                'filename': filepath.name,
                'filepath': str(filepath),
                'text': text,
                'section': section,
                'data_type': 'final_dossier'
            })
    
    return records

def process_guidelines(guidelines_path):
    """Process health authority guidelines"""
    records = []
    base = Path(guidelines_path)
    
    if not base.exists():
        print(f"Guidelines path not found: {guidelines_path}")
        return records
    
    extensions = ['*.pdf', '*.docx', '*.doc']
    all_files = []
    for ext in extensions:
        all_files.extend(base.rglob(ext))
    
    print(f"Found {len(all_files)} guideline files")
    
    # Map guideline types to CTD sections
    guideline_mappings = {
        'ich m4q': ['3.2.P.1', '3.2.P.2', '3.2.P.3', '3.2.P.4', '3.2.P.5', '3.2.P.7', '3.2.P.8'],
        'ich q1': ['3.2.P.8'],  # Stability
        'ich q2': ['3.2.P.5'],  # Analytical validation
        'ich q3': ['3.2.P.5'],  # Impurities
        'ich q6': ['3.2.P.5'],  # Specifications
        'ich q8': ['3.2.P.2'],  # Pharmaceutical development
        'ich q9': ['3.2.P.2'],  # Quality risk management
        'ich q11': ['3.2.S.2'],  # Drug substance
        'ich m4e': ['2.5', '2.7', '5.3.1'],  # Efficacy
        'ich m4s': ['2.4', '2.6'],  # Safety
        'bioequivalence': ['5.3.1'],
        'bioavailability': ['5.3.1'],
        'stability': ['3.2.P.8'],
        'dissolution': ['3.2.P.2.2', '3.2.P.5'],
        'excipient': ['3.2.P.4'],
        'container closure': ['3.2.P.7'],
        'manufacturing': ['3.2.P.3'],
    }
    
    for filepath in tqdm(all_files, desc="Extracting guidelines"):
        filename_lower = filepath.name.lower()
        
        # Find matching sections
        matched_sections = []
        for keyword, sections in guideline_mappings.items():
            if keyword in filename_lower or keyword in str(filepath).lower():
                matched_sections.extend(sections)
        
        if not matched_sections:
            continue
        
        ext = filepath.suffix.lower()
        if ext == '.pdf':
            text = extract_pdf_text(str(filepath))
        elif ext in ['.docx', '.doc']:
            text = extract_docx_text(str(filepath))
        else:
            continue
        
        if len(text) < 100:
            continue
        
        # Add to each matched section
        for section in set(matched_sections):
            records.append({
                'filename': filepath.name,
                'filepath': str(filepath),
                'text': text[:8000],  # Slightly shorter for guidelines
                'section': section,
                'data_type': 'guideline'
            })
    
    return records

if __name__ == "__main__":
    # ============================================
    # CONFIGURE YOUR PATHS HERE
    # ============================================
    
    # Path to source documents (study reports, CoAs, validation reports, etc.)
    SOURCE_PATH = r"C:\Users\GudjonAsmundsson\OneDrive - Sagareg\Documents\SagaReg\Apixaban\OneDrive_2_4-29-2025"
    
    # Path to final dossier chapters (the actual CTD documents)
    FINAL_DOSSIER_PATH = r"C:\Users\GudjonAsmundsson\OneDrive - Sagareg\Documents\SagaReg\Apixaban\OneDrive_2_4-29-2025\Module 3\Final module 3 part of dossier Apixaban"
    
    # Path to health authority guidelines (optional - set to None if not available)
    GUIDELINES_PATH = r"C:\Users\GudjonAsmundsson\OneDrive - Sagareg\Documents\SagaReg\Guidelines"
    # GUIDELINES_PATH = None  # Uncomment if no guidelines available
    
    # ============================================
    # EXTRACTION
    # ============================================
    
    all_records = []
    
    # 1. Source documents
    print("\n" + "="*50)
    print("EXTRACTING SOURCE DOCUMENTS")
    print("="*50)
    source_records = process_source_documents(SOURCE_PATH)
    all_records.extend(source_records)
    print(f"Extracted {len(source_records)} source documents")
    
    # 2. Final dossier
    print("\n" + "="*50)
    print("EXTRACTING FINAL DOSSIER")
    print("="*50)
    if FINAL_DOSSIER_PATH:
        dossier_records = process_final_dossier(FINAL_DOSSIER_PATH)
        all_records.extend(dossier_records)
        print(f"Extracted {len(dossier_records)} final dossier chunks")
    
    # 3. Guidelines
    print("\n" + "="*50)
    print("EXTRACTING GUIDELINES")
    print("="*50)
    if GUIDELINES_PATH:
        guideline_records = process_guidelines(GUIDELINES_PATH)
        all_records.extend(guideline_records)
        print(f"Extracted {len(guideline_records)} guideline entries")
    
    # Create dataframe
    df = pd.DataFrame(all_records)
    
    # Add hierarchy columns
    def get_module(section):
        if section.startswith('2.'):
            return 'Module2'
        elif section.startswith('3.'):
            return 'Module3'
        elif section.startswith('5.'):
            return 'Module5'
        return 'Other'
    
    df['module'] = df['section'].apply(get_module)
    
    # Summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"\nTotal documents: {len(df)}")
    print(f"\nBy data type:")
    print(df['data_type'].value_counts())
    print(f"\nBy section:")
    print(df['section'].value_counts())
    print(f"\nBy module:")
    print(df['module'].value_counts())
    
    # Save
    df.to_csv("data/processed/documents.csv", index=False)
    print(f"\nSaved to data/processed/documents.csv")
