import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Benchmark results from full 3,045 queries evaluation on NVIDIA A100 GPU
# Architecture: Hybrid Retrieval (BGE-M3 + BM25 via RRF) + BGE-Reranker-Large + Qwen 2.5 14B BF16
data = [
    {"Category": "OVERALL", "Count": 3045, "Hit@1": 95.3, "Hit@6": 99.3, "MRR": 0.970, "Token F1": 0.416, "Semantic F1": 0.784, "ROUGE-L": 0.367, "Avg Latency (ms)": 867.5},
    {"Category": "Type: abstractive", "Count": 1793, "Hit@1": 94.9, "Hit@6": 99.2, "MRR": 0.966, "Token F1": 0.385, "Semantic F1": 0.820, "ROUGE-L": 0.318, "Avg Latency (ms)": 873.9},
    {"Category": "Type: extractive", "Count": 1252, "Hit@1": 96.0, "Hit@6": 99.4, "MRR": 0.975, "Token F1": 0.460, "Semantic F1": 0.733, "ROUGE-L": 0.437, "Avg Latency (ms)": 858.3},
    {"Category": "Source: text", "Count": 1914, "Hit@1": 96.7, "Hit@6": 99.7, "MRR": 0.979, "Token F1": 0.455, "Semantic F1": 0.786, "ROUGE-L": 0.412, "Avg Latency (ms)": 866.8},
    {"Category": "Source: text-table", "Count": 148, "Hit@1": 95.9, "Hit@6": 99.3, "MRR": 0.974, "Token F1": 0.374, "Semantic F1": 0.776, "ROUGE-L": 0.310, "Avg Latency (ms)": 857.8},
    {"Category": "Source: text-image", "Count": 763, "Hit@1": 92.4, "Hit@6": 98.4, "MRR": 0.949, "Token F1": 0.347, "Semantic F1": 0.786, "ROUGE-L": 0.287, "Avg Latency (ms)": 874.2},
    {"Category": "Source: text-table-image", "Count": 220, "Hit@1": 93.6, "Hit@6": 98.6, "MRR": 0.957, "Token F1": 0.345, "Semantic F1": 0.768, "ROUGE-L": 0.288, "Avg Latency (ms)": 856.4},
]

df = pd.DataFrame(data)
os.makedirs("sample_data", exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
categories = [c.replace("Type: ", "").replace("Source: ", "") for c in df["Category"]]
y_pos = np.arange(len(categories))

# ==============================================================================
# PLOT 1: RETRIEVAL ACCURACY (Hit@1 vs Hit@6)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(11, 6.5), dpi=300)
width = 0.36
rects1 = ax1.barh(y_pos - width/2, df["Hit@1"], width, label='Hit@1 (Top-1 Exact Document)', color='#4338CA', alpha=0.92)
rects2 = ax1.barh(y_pos + width/2, df["Hit@6"], width, label='Hit@6 (Top-6 Candidate Window)', color='#818CF8', alpha=0.88)

ax1.set_xlabel('Retrieval Accuracy (%)', fontsize=12, fontweight='bold', color='#0F172A', labelpad=10)
ax1.set_title('Figure 1: Retrieval Accuracy by Question Category & Modality\n(Hybrid Dense BGE-M3 + BM25 + BGE-Reranker-Large on NVIDIA A100)', 
              fontsize=13, fontweight='bold', color='#1E1B4B', pad=15)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(categories, fontsize=11, fontweight='medium', color='#1E293B')
ax1.set_xlim(80, 103)
ax1.legend(loc='lower right', frameon=True, fontsize=10, facecolor='#F8FAFC', edgecolor='#CBD5E1')
ax1.grid(axis='x', linestyle='--', alpha=0.6)

for r in rects1:
    ax1.annotate(f"{r.get_width():.1f}%", xy=(r.get_width() + 0.3, r.get_y() + r.get_height()/2),
                 va='center', fontsize=9.5, color='#1E1B4B', fontweight='bold')
for r in rects2:
    ax1.annotate(f"{r.get_width():.1f}%", xy=(r.get_width() + 0.3, r.get_y() + r.get_height()/2),
                 va='center', fontsize=9, color='#475569')

plt.tight_layout()
p1_path = "sample_data/plot1_retrieval_accuracy.png"
fig1.savefig(p1_path, dpi=300)
plt.close(fig1)
print(f"[SAVED] {p1_path}")

# ==============================================================================
# PLOT 2: GENERATION QUALITY (Semantic F1, Token F1, ROUGE-L)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(11, 6.5), dpi=300)
w3 = 0.26
rects_tok = ax2.barh(y_pos - w3, df["Token F1"], w3, label='Token F1 (SQuAD Exact Match)', color='#F59E0B', alpha=0.92)
rects_sem = ax2.barh(y_pos, df["Semantic F1"], w3, label='Semantic F1 (BGE-M3 Cosine Similarity)', color='#10B981', alpha=0.92)
rects_rg  = ax2.barh(y_pos + w3, df["ROUGE-L"], w3, label='ROUGE-L (Longest Common Subsequence)', color='#6366F1', alpha=0.85)

ax2.set_xlabel('Evaluation Score [0.0 - 1.0]', fontsize=12, fontweight='bold', color='#0F172A', labelpad=10)
ax2.set_title('Figure 2: Generation Quality Metrics across Question Categories\n(Question-Anchored Declarative Synthesis with Qwen 2.5 14B BF16)', 
              fontsize=13, fontweight='bold', color='#1E1B4B', pad=15)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(categories, fontsize=11, fontweight='medium', color='#1E293B')
ax2.set_xlim(0.0, 1.05)
ax2.axvline(0.8, color='#EF4444', linestyle='--', linewidth=1.8, label='Target Benchmark Threshold (0.80)')
ax2.legend(loc='lower right', frameon=True, fontsize=10, facecolor='#F8FAFC', edgecolor='#CBD5E1')
ax2.grid(axis='x', linestyle='--', alpha=0.6)

for r in rects_sem:
    ax2.annotate(f"{r.get_width():.3f}", xy=(r.get_width() + 0.015, r.get_y() + r.get_height()/2),
                 va='center', fontsize=9, color='#065F46', fontweight='bold')
for r in rects_tok:
    ax2.annotate(f"{r.get_width():.3f}", xy=(r.get_width() + 0.015, r.get_y() + r.get_height()/2),
                 va='center', fontsize=8.5, color='#92400E')

plt.tight_layout()
p2_path = "sample_data/plot2_generation_quality.png"
fig2.savefig(p2_path, dpi=300)
plt.close(fig2)
print(f"[SAVED] {p2_path}")

# ==============================================================================
# PLOT 3: MODALITY SOURCE BREAKDOWN
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
modality_rows = df[df["Category"].str.contains("Source:|OVERALL")].copy()
m_cats = [c.replace("Source: ", "") for c in modality_rows["Category"]]
m_pos = np.arange(len(m_cats))

b1 = ax3.bar(m_pos - 0.25, modality_rows["Semantic F1"], 0.24, label='Semantic F1 (Cos Sim)', color='#10B981', alpha=0.92)
b2 = ax3.bar(m_pos, modality_rows["Token F1"], 0.24, label='Token F1 (Lexical)', color='#F59E0B', alpha=0.92)
b3 = ax3.bar(m_pos + 0.25, modality_rows["ROUGE-L"], 0.24, label='ROUGE-L (Recall)', color='#6366F1', alpha=0.88)

ax3.set_ylabel('Metric Score', fontsize=12, fontweight='bold', color='#0F172A', labelpad=10)
ax3.set_title('Figure 3: Generation Performance by Document Modality Source\n(Plain Text vs. Tables, Figures, and Multimodal Layouts)', 
              fontsize=13, fontweight='bold', color='#1E1B4B', pad=15)
ax3.set_xticks(m_pos)
ax3.set_xticklabels(m_cats, fontsize=11, fontweight='medium', color='#1E293B')
ax3.set_ylim(0.0, 1.05)
ax3.axhline(0.8, color='#EF4444', linestyle='--', linewidth=1.5, alpha=0.7, label='Benchmark Target (0.80)')
ax3.legend(loc='upper right', frameon=True, fontsize=10, facecolor='#F8FAFC', edgecolor='#CBD5E1')
ax3.grid(axis='y', linestyle='--', alpha=0.6)

for b in b1:
    ax3.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height() + 0.02),
                 ha='center', fontsize=9, fontweight='bold', color='#065F46')
for b in b2:
    ax3.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height() + 0.02),
                 ha='center', fontsize=8.5, color='#92400E')

plt.tight_layout()
p3_path = "sample_data/plot3_modality_breakdown.png"
fig3.savefig(p3_path, dpi=300)
plt.close(fig3)
print(f"[SAVED] {p3_path}")

# ==============================================================================
# PLOT 4: LATENCY PROFILE
# ==============================================================================
fig4, ax4 = plt.subplots(figsize=(11, 5.5), dpi=300)
bars = ax4.bar(y_pos, df["Avg Latency (ms)"], color='#0284C7', alpha=0.88, width=0.55, edgecolor='#0369A1', linewidth=1)
ax4.set_ylabel('Latency per Query (ms)', fontsize=12, fontweight='bold', color='#0F172A', labelpad=10)
ax4.set_title('Figure 4: End-to-End Latency Profile per Query Category\n(Batched Execution with Batch Size = 8 on NVIDIA A100)', 
              fontsize=13, fontweight='bold', color='#1E1B4B', pad=15)
ax4.set_xticks(y_pos)
ax4.set_xticklabels(categories, fontsize=10.5, rotation=20, ha='right', fontweight='medium', color='#1E293B')
ax4.set_ylim(800, 1050)
mean_lat = df["Avg Latency (ms)"].mean()
ax4.axhline(mean_lat, color='#DC2626', linestyle='--', linewidth=1.6, label=f'Mean System Latency: {mean_lat:.1f} ms')
ax4.legend(loc='upper right', frameon=True, fontsize=10, facecolor='#F8FAFC', edgecolor='#CBD5E1')
ax4.grid(axis='y', linestyle='--', alpha=0.6)

for b in bars:
    ax4.annotate(f"{b.get_height():.1f} ms", (b.get_x() + b.get_width()/2, b.get_height() + 6),
                 ha='center', fontsize=9, fontweight='bold', color='#0C4A6E')

plt.tight_layout()
p4_path = "sample_data/plot4_latency_profile.png"
fig4.savefig(p4_path, dpi=300)
plt.close(fig4)
print(f"[SAVED] {p4_path}")

# Also generate the combined composite 4-panel overview for reference
fig_all, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=250)
fig_all.suptitle("Vectara Open RAG Benchmark - Full 3,045 Queries Evaluation\n(NVIDIA A100 GPU | BGE-M3 + BM25 + BGE-Reranker + Qwen 2.5 14B BF16)", 
                 fontsize=15, fontweight='bold', color='#0F172A', y=0.98)

# Panel 1
ax1 = axes[0, 0]
rects1 = ax1.barh(y_pos - width/2, df["Hit@1"], width, label='Hit@1 (%)', color='#4338CA', alpha=0.9)
rects2 = ax1.barh(y_pos + width/2, df["Hit@6"], width, label='Hit@6 (%)', color='#818CF8', alpha=0.85)
ax1.set_xlabel('Retrieval Accuracy (%)', fontweight='bold')
ax1.set_title('1. Retrieval Accuracy (Hit@1 vs Hit@6)', fontsize=12, fontweight='bold', color='#4338CA')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(categories)
ax1.set_xlim(80, 103)
ax1.legend(loc='lower right', frameon=True, fontsize=9)
ax1.grid(axis='x', linestyle='--', alpha=0.6)
for r in rects1:
    ax1.annotate(f"{r.get_width():.1f}%", (r.get_width() + 0.3, r.get_y() + r.get_height()/2), va='center', fontsize=8, fontweight='bold')

# Panel 2
ax2 = axes[0, 1]
ax2.barh(y_pos - w3, df["Token F1"], w3, label='Token F1', color='#F59E0B', alpha=0.9)
ax2.barh(y_pos, df["Semantic F1"], w3, label='Semantic F1', color='#10B981', alpha=0.9)
ax2.barh(y_pos + w3, df["ROUGE-L"], w3, label='ROUGE-L', color='#6366F1', alpha=0.8)
ax2.set_xlabel('Metric Score', fontweight='bold')
ax2.set_title('2. Generation Quality (Semantic F1 vs Token F1 & ROUGE-L)', fontsize=12, fontweight='bold', color='#4338CA')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(categories)
ax2.set_xlim(0.0, 1.05)
ax2.axvline(0.8, color='#EF4444', linestyle='--', linewidth=1.5, label='Benchmark Target (0.80)')
ax2.legend(loc='lower right', frameon=True, fontsize=9)
ax2.grid(axis='x', linestyle='--', alpha=0.6)

# Panel 3
ax3 = axes[1, 0]
ax3.bar(m_pos - 0.25, modality_rows["Semantic F1"], 0.24, label='Semantic F1', color='#10B981')
ax3.bar(m_pos, modality_rows["Token F1"], 0.24, label='Token F1', color='#F59E0B')
ax3.bar(m_pos + 0.25, modality_rows["ROUGE-L"], 0.24, label='ROUGE-L', color='#6366F1')
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('3. Quality by Document Modality Source', fontsize=12, fontweight='bold', color='#4338CA')
ax3.set_xticks(m_pos)
ax3.set_xticklabels(m_cats, rotation=15, ha='right')
ax3.set_ylim(0.0, 1.05)
ax3.legend(loc='upper right', frameon=True, fontsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.6)

# Panel 4
ax4 = axes[1, 1]
b_lat = ax4.bar(y_pos, df["Avg Latency (ms)"], color='#0284C7', alpha=0.85, width=0.55)
ax4.set_ylabel('Latency (ms)', fontweight='bold')
ax4.set_title('4. End-to-End Latency per Query (A100 Batch Size 8)', fontsize=12, fontweight='bold', color='#4338CA')
ax4.set_xticks(y_pos)
ax4.set_xticklabels(categories, rotation=20, ha='right')
ax4.set_ylim(800, 1050)
ax4.axhline(mean_lat, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Mean: {mean_lat:.1f} ms')
ax4.legend(loc='upper right', frameon=True, fontsize=9)
ax4.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
combined_path = "sample_data/benchmark_results.png"
fig_all.savefig(combined_path, dpi=300)
plt.close(fig_all)
print(f"[SAVED] {combined_path}")
print("[DONE] All 4 individual plots and composite benchmark dashboard generated successfully!")
