import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
input_color = '#e8f4f8'
encoder_color = '#d4edda'
head_color = '#fff3cd'
output_color = '#f8d7da'
arrow_color = '#495057'

def draw_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

# Input
draw_box(ax, 1, 7.5, 3, 1.5, 'Document\nText', input_color, 11)

# BioBERT
draw_box(ax, 1, 5, 3, 1.5, 'BioBERT\nEmbedding\n(768-dim)', '#cce5ff', 10)

# Shared Encoder
draw_box(ax, 5.5, 5, 3, 2, 'Shared Encoder\n\nLinear(768→256)\nBatchNorm + ReLU\nDropout(0.3)\nLinear(256→256)\nBatchNorm + ReLU', encoder_color, 9)

# Level 1 Head
draw_box(ax, 10, 7, 3, 1.5, 'Level 1 Head\nLinear(256→3)\n\nModule Classification', head_color, 9)

# Level 2 Head
draw_box(ax, 10, 4, 3, 2, 'Level 2 Head\nLinear(256+3→10)\n\nSection Classification\n(Conditioned on L1)', head_color, 9)

# Outputs
draw_box(ax, 10, 1, 3, 1.5, 'Predictions\n\nModule: 98.4%\nSection: 74.9%', output_color, 9)

# Arrows
draw_arrow(ax, (2.5, 7.5), (2.5, 6.5))  # Input to BioBERT
draw_arrow(ax, (4, 5.75), (5.5, 5.75))  # BioBERT to Encoder
draw_arrow(ax, (8.5, 6.5), (10, 7.5))   # Encoder to L1
draw_arrow(ax, (8.5, 5.5), (10, 5.5))   # Encoder to L2
draw_arrow(ax, (11.5, 7), (11.5, 6))    # L1 to L2 (conditioning)
draw_arrow(ax, (11.5, 4), (11.5, 2.5))  # L2 to output

# Conditioning arrow label
ax.text(12, 6.5, 'L1 probs\nconcatenated', fontsize=8, style='italic')

# Title
ax.text(7, 9.5, 'Hierarchical CTD Document Classifier Architecture', 
        ha='center', fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input'),
    mpatches.Patch(facecolor='#cce5ff', edgecolor='black', label='Pre-trained Embeddings'),
    mpatches.Patch(facecolor=encoder_color, edgecolor='black', label='Trainable Encoder'),
    mpatches.Patch(facecolor=head_color, edgecolor='black', label='Classification Heads'),
    mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig('figures/architecture.png', dpi=150, bbox_inches='tight')
print("Saved figures/architecture.png")
plt.show()