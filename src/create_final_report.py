from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc = Document()

# Title
title = doc.add_heading('Hierarchical Deep Learning for Automated CTD Document Classification in Pharmaceutical Regulatory Submissions', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('CS230 Final Project Report')
doc.add_paragraph('Authors: [Your Names]')
doc.add_paragraph('Date: December 2024')
doc.add_paragraph()

# Abstract
doc.add_heading('Abstract', level=1)
doc.add_paragraph(
    'Pharmaceutical regulatory submissions require organizing hundreds of source documents into a standardized '
    'Common Technical Document (CTD) structure. Current approaches rely on keyword matching, achieving only 68% '
    'accuracy and requiring extensive manual review. We propose a hierarchical classification approach using '
    'BioBERT embeddings that predicts document-to-section mappings at multiple granularity levels (Module → Section). '
    'Our model achieves 74.9% accuracy on fine-grained section classification and 98.4% on module-level classification, '
    'representing a 7.1 percentage point improvement over baselines.'
)

# 1. Introduction
doc.add_heading('1. Introduction', level=1)
doc.add_paragraph(
    'The pharmaceutical industry faces a significant challenge in regulatory submissions: organizing thousands of '
    'source documents into the ICH Common Technical Document (CTD) format. This process currently requires weeks '
    'of manual effort from regulatory professionals, with costs ranging from $500K to $2M per submission.'
)
doc.add_paragraph(
    'The Problem: When generating regulatory dossiers using Large Language Models (LLMs), the system must know '
    'which source documents contain relevant information for each CTD section. Incorrect document-to-section '
    'mapping leads to generated content that cites wrong sources or misses critical data.'
)
doc.add_paragraph(
    'Our Contribution: We present a hierarchical neural network classifier that: (1) Maps pharmaceutical documents '
    'to CTD sections with 74.9% accuracy, (2) Achieves near-perfect module-level classification (98.4%), '
    '(3) Outperforms TF-IDF baselines by 7.1 percentage points, (4) Enables automated document routing for LLM-based dossier generation.'
)

# 2. Related Work
doc.add_heading('2. Related Work', level=1)
doc.add_paragraph(
    'Document Classification: Transformer-based models have achieved state-of-the-art results on document '
    'classification tasks. BERT and its variants provide contextual embeddings that capture semantic meaning '
    'beyond simple keyword matching.'
)
doc.add_paragraph(
    'Biomedical NLP: BioBERT (Lee et al., 2020) pre-trained on PubMed abstracts has shown superior performance '
    'on biomedical text mining tasks, including named entity recognition and relation extraction.'
)
doc.add_paragraph(
    'Hierarchical Classification: Multi-level classification approaches exploit label hierarchies to improve '
    'predictions, particularly for fine-grained categories with limited training data.'
)

# 3. Dataset
doc.add_heading('3. Dataset', level=1)

doc.add_heading('3.1 Data Source', level=2)
doc.add_paragraph(
    'We use documents from an Apixaban (anticoagulant drug) regulatory dossier, including: '
    'Clinical literature references (Module 2), Quality/CMC documentation (Module 3), and Clinical study reports (Module 5).'
)

doc.add_heading('3.2 Statistics', level=2)
table = doc.add_table(rows=5, cols=2)
table.style = 'Table Grid'
data = [('Metric', 'Value'), ('Total Documents', '912'), ('CTD Sections (classes)', '10'), 
        ('CTD Modules', '3'), ('Train/Test Split', '729/183 (80/20)')]
for i, (c1, c2) in enumerate(data):
    table.rows[i].cells[0].text = c1
    table.rows[i].cells[1].text = c2

doc.add_paragraph()
doc.add_heading('3.3 Class Distribution', level=2)

# Add class distribution figure
if os.path.exists('figures/class_distribution.png'):
    doc.add_picture('figures/class_distribution.png', width=Inches(5.5))
    doc.add_paragraph('Figure 1: Document distribution across CTD sections', style='Caption')
else:
    doc.add_paragraph('[Figure 1: class_distribution.png - FILE NOT FOUND]')

doc.add_paragraph(
    'The dataset exhibits significant class imbalance, with clinical literature (Sections 2.4 and 2.5) comprising 83% of documents.'
)

# 4. Methods
doc.add_heading('4. Methods', level=1)

doc.add_heading('4.1 Document Representation', level=2)
doc.add_paragraph(
    'We extract text from PDF and Word documents using PyMuPDF and python-docx. Each document is truncated to '
    '10,000 characters. Documents are embedded using BioBERT (dmis-lab/biobert-base-cased-v1.2), producing '
    '768-dimensional vectors that capture biomedical semantic content.'
)

doc.add_heading('4.2 Hierarchical Architecture', level=2)
doc.add_paragraph(
    'Our model employs a two-level hierarchical approach: Level 1 predicts the CTD module (2, 3, or 5) with 3 '
    'output classes. Level 2 predicts the specific CTD section with 10 output classes, conditioned on Level 1 predictions.'
)

# Add architecture figure
if os.path.exists('figures/architecture.png'):
    doc.add_picture('figures/architecture.png', width=Inches(5.5))
    doc.add_paragraph('Figure 2: Hierarchical classifier architecture', style='Caption')
else:
    doc.add_paragraph('[Figure 2: architecture.png - FILE NOT FOUND]')

doc.add_paragraph('The architecture consists of:')
doc.add_paragraph('• Shared Encoder: Linear(768→256) + BatchNorm + ReLU + Dropout(0.3) × 2 layers')
doc.add_paragraph('• Level 1 Head: Linear(256→3) for module classification')
doc.add_paragraph('• Level 2 Head: Linear(256+3→10), concatenates shared features with Level 1 probabilities')

doc.add_heading('4.3 Hierarchical Loss Function', level=2)
doc.add_paragraph('We use a weighted combination: L = 0.3 × L_level1 + 0.7 × L_level2')
doc.add_paragraph('The higher weight on Level 2 emphasizes fine-grained classification while benefiting from hierarchical structure.')

doc.add_heading('4.4 Training Details', level=2)
table = doc.add_table(rows=8, cols=2)
table.style = 'Table Grid'
data = [('Hyperparameter', 'Value'), ('Optimizer', 'Adam'), ('Learning Rate', '1e-3'), 
        ('Weight Decay', '1e-5'), ('Batch Size', '16'), ('Epochs', '50'), 
        ('Dropout', '0.3'), ('LR Scheduler', 'ReduceLROnPlateau (patience=5)')]
for i, (c1, c2) in enumerate(data):
    table.rows[i].cells[0].text = c1
    table.rows[i].cells[1].text = c2

# 5. Results
doc.add_paragraph()
doc.add_heading('5. Results', level=1)

doc.add_heading('5.1 Main Results', level=2)

# Results table
table = doc.add_table(rows=7, cols=3)
table.style = 'Table Grid'
data = [
    ('Method', 'Accuracy', 'Improvement'),
    ('TF-IDF + Logistic Regression', '67.76%', 'baseline'),
    ('BioBERT + MLP', '68.31%', '+0.6%'),
    ('BioBERT + Logistic Regression', '69.95%', '+2.2%'),
    ('BERT (general) + LogReg', '71.58%', '+3.8%'),
    ('BioBERT + SVM', '73.77%', '+6.0%'),
    ('Hierarchical NN (Ours)', '74.86%', '+7.1%'),
]
for i, row_data in enumerate(data):
    for j, val in enumerate(row_data):
        table.rows[i].cells[j].text = val

doc.add_paragraph()

# Add model comparison figure
if os.path.exists('figures/model_comparison.png'):
    doc.add_picture('figures/model_comparison.png', width=Inches(5))
    doc.add_paragraph('Figure 3: Model comparison showing accuracy improvements', style='Caption')

doc.add_heading('5.2 Hierarchical Performance', level=2)
doc.add_paragraph('• Module Level (3 classes): 98.36% accuracy')
doc.add_paragraph('• Section Level (10 classes): 74.86% accuracy')
doc.add_paragraph(
    'The near-perfect module-level accuracy demonstrates the effectiveness of the hierarchical approach.'
)

doc.add_heading('5.3 Training Dynamics', level=2)
# Add training curves
if os.path.exists('figures/training_curves.png'):
    doc.add_picture('figures/training_curves.png', width=Inches(5.5))
    doc.add_paragraph('Figure 4: Training loss and validation accuracy over epochs', style='Caption')

doc.add_heading('5.4 Confusion Matrix', level=2)
# Add confusion matrix
if os.path.exists('figures/confusion_matrix.png'):
    doc.add_picture('figures/confusion_matrix.png', width=Inches(4.5))
    doc.add_paragraph('Figure 5: Confusion matrix for section-level predictions', style='Caption')

doc.add_paragraph(
    'The model performs well on sections with sufficient training data (2.5, 5.3.1) but struggles with '
    'extremely rare classes (3.2.P.2, 3.2.P.3) that have fewer than 5 training examples.'
)

doc.add_heading('5.5 Ablation Studies', level=2)
doc.add_paragraph('Effect of Classifier Architecture:')
table = doc.add_table(rows=5, cols=2)
table.style = 'Table Grid'
data = [('Classifier', 'Accuracy'), ('Logistic Regression', '69.95%'), 
        ('MLP (256, 128)', '68.31%'), ('SVM (RBF)', '73.77%'), ('Hierarchical NN', '74.86%')]
for i, (c1, c2) in enumerate(data):
    table.rows[i].cells[0].text = c1
    table.rows[i].cells[1].text = c2

# 6. Discussion
doc.add_paragraph()
doc.add_heading('6. Discussion', level=1)

doc.add_heading('6.1 Key Findings', level=2)
doc.add_paragraph('1. Hierarchical classification improves performance by first predicting the module (98.4% accuracy), providing context for section classification.')
doc.add_paragraph('2. Deep learning outperforms traditional methods with 7.1% improvement over TF-IDF baseline.')
doc.add_paragraph('3. Class imbalance remains challenging for sections with fewer than 5 training examples.')

doc.add_heading('6.2 Limitations', level=2)
doc.add_paragraph('• Small dataset: 912 documents from single drug product limits generalization')
doc.add_paragraph('• Class imbalance: 65% of data belongs to one class (2.5)')
doc.add_paragraph('• PDF extraction quality: Some documents yield poor text extraction')

doc.add_heading('6.3 Future Work', level=2)
doc.add_paragraph('1. Data augmentation for rare classes')
doc.add_paragraph('2. Multi-dossier training across drug products')
doc.add_paragraph('3. Fine-tuning BioBERT end-to-end')
doc.add_paragraph('4. Production deployment at SagaReg')

# 7. Conclusion
doc.add_heading('7. Conclusion', level=1)
doc.add_paragraph(
    'We presented a hierarchical deep learning approach for classifying pharmaceutical regulatory documents '
    'into CTD sections. Our model achieves 74.9% accuracy, outperforming TF-IDF baselines by 7.1 percentage points. '
    'The hierarchical structure proves particularly effective, achieving 98.4% accuracy at the module level. '
    'This classifier enables automated document routing for LLM-powered dossier generation.'
)

# References
doc.add_heading('References', level=1)
doc.add_paragraph('1. Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model. Bioinformatics.')
doc.add_paragraph('2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.')
doc.add_paragraph('3. ICH M4 (2016). Organisation of the Common Technical Document.')

# Save
doc.save('report/cs230_report_with_figures.docx')
print('Saved report/cs230_report_with_figures.docx')
print('\nOpen in Word → File → Save As → PDF')