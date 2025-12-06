import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/processed/documents.csv")

# Class distribution
plt.figure(figsize=(12, 6))
section_counts = df['section'].value_counts()

colors = ['#2ecc71' if 'Module' not in str(s) else '#3498db' for s in section_counts.index]
bars = plt.bar(range(len(section_counts)), section_counts.values, color=colors)

plt.xticks(range(len(section_counts)), section_counts.index, rotation=45, ha='right')
plt.xlabel('CTD Section', fontsize=12)
plt.ylabel('Number of Documents', fontsize=12)
plt.title('Document Distribution Across CTD Sections', fontsize=14)

# Add value labels on bars
for bar, count in zip(bars, section_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight')
print("Saved figures/class_distribution.png")
plt.show()

# Also create a pie chart for modules
plt.figure(figsize=(8, 8))
module_counts = df['section'].apply(lambda x: 'Module 2' if x.startswith('2.') 
                                    else ('Module 3' if x.startswith('3.') else 'Module 5')).value_counts()

colors = ['#3498db', '#2ecc71', '#e74c3c']
plt.pie(module_counts.values, labels=module_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12})
plt.title('Document Distribution by CTD Module', fontsize=14)
plt.tight_layout()
plt.savefig('figures/module_distribution.png', dpi=150)
print("Saved figures/module_distribution.png")
plt.show()