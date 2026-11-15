from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


project_root = Path(__file__).resolve().parent.parent

target_dir = project_root / "plots"
target_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(project_root / "data" / "cleaned_form_responses.csv")

person_palette = {
    "P1": "#4E79A7",
    "P2": "#F28E2B",
    "P3": "#59A14F",
    "P4": "#E15759",
    "P5": "#B07AA1",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#ebebeb",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})

fig1, axes1 = plt.subplots(2, 2, figsize=(9, 7))

sns.histplot(
    df["wpm_avg"], kde=True, ax=axes1[0, 0], color="#4a6fe3", edgecolor="black"
)
axes1[0, 0].set_title("Distribution of Typing Speed (WPM)")
axes1[0, 0].set_xlabel("Average WPM")
axes1[0, 0].set_ylabel("Frequency")

sns.histplot(
    df["acc_avg"], kde=True, ax=axes1[0, 1], color="#2ca02c", edgecolor="black"
)
axes1[0, 1].set_title("Distribution of Typing Accuracy (%)")
axes1[0, 1].set_xlabel("Average Accuracy (%)")
axes1[0, 1].set_ylabel("Frequency")

sns.histplot(
    df["sleep_hours"],
    discrete=True,
    ax=axes1[1, 0],
    color="#ff7f0e",
    edgecolor="black",
)
axes1[1, 0].set_title("Distribution of Sleep Duration")
axes1[1, 0].set_xlabel("Sleep Hours")
axes1[1, 0].set_ylabel("Frequency")

sns.countplot(
    data=df,
    x="stress",
    order=["low", "medium", "high"],
    ax=axes1[1, 1],
    color="#6c757d",
    edgecolor="black",
)
axes1[1, 1].set_title("Stress Level Frequency")
axes1[1, 1].set_xlabel("Stress Level")
axes1[1, 1].set_ylabel("Count")

plt.tight_layout()
fig1.savefig(target_dir / "histograms.png", dpi=300)
plt.close(fig1)

fig2, axes2 = plt.subplots(2, 2, figsize=(9.5, 7.5))

sns.boxplot(
    data=df,
    x="person_id",
    y="wpm_avg",
    ax=axes2[0, 0],
    hue="person_id",
    palette=person_palette,
    width=0.45,
)
axes2[0, 0].set_title("Typing Speed by Participant")
axes2[0, 0].set_xlabel("Participant ID")
axes2[0, 0].set_ylabel("Average WPM")

sns.boxplot(
    data=df,
    x="person_id",
    y="acc_avg",
    ax=axes2[0, 1],
    hue="person_id",
    palette=person_palette,
    width=0.45,
)
axes2[0, 1].set_title("Typing Accuracy by Participant")
axes2[0, 1].set_xlabel("Participant ID")
axes2[0, 1].set_ylabel("Average Accuracy (%)")

sns.boxplot(
    data=df,
    x="caffeine_mg",
    y="wpm_avg",
    ax=axes2[1, 0],
    color="#d0d7de",
    width=0.45,
)
axes2[1, 0].set_title("Typing Speed by Caffeine Dose")
axes2[1, 0].set_xlabel("Caffeine (mg)")
axes2[1, 0].set_ylabel("Average WPM")

sns.boxplot(
    data=df,
    x="stress",
    y="wpm_avg",
    order=["low", "medium", "high"],
    ax=axes2[1, 1],
    color="#d0d7de",
    width=0.45,
)
axes2[1, 1].set_title("Typing Speed by Stress Level")
axes2[1, 1].set_xlabel("Stress Level")
axes2[1, 1].set_ylabel("Average WPM")

plt.tight_layout()
fig2.savefig(target_dir / "boxplots.png", dpi=300)
plt.close(fig2)

fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5))

num_cols = [
    "sleep_hours",
    "sleep_quality",
    "caffeine_mg",
    "wpm_avg",
    "acc_avg",
]
corr = df[num_cols].corr(method="pearson")
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=-1,
    vmax=1,
    square=True,
    ax=axes3[0],
    cbar_kws={"shrink": 0.8},
)
axes3[0].set_title("Pearson Correlation Matrix")

sns.scatterplot(
    data=df,
    x="sleep_hours",
    y="wpm_avg",
    hue="person_id",
    s=55,
    palette=person_palette,
    ax=axes3[1],
)
sns.regplot(
    data=df,
    x="sleep_hours",
    y="wpm_avg",
    scatter=False,
    color="#333333",
    line_kws={"ls": "--", "lw": 1.2},
    ax=axes3[1],
)
axes3[1].set_title("Typing Speed vs. Sleep Duration")
axes3[1].set_xlabel("Sleep Hours")
axes3[1].set_ylabel("Average WPM")
axes3[1].legend(title="Participant", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
fig3.savefig(target_dir / "pearson_scatter.png", dpi=300)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(9, 4))
sns.lineplot(
    data=df,
    x="day_number",
    y="wpm_avg",
    hue="person_id",
    marker="o",
    palette=person_palette,
    ax=ax4,
)
ax4.set_title("Typing Speed Progression Across Study Days")
ax4.set_xlabel("Chronological Day Number")
ax4.set_ylabel("Average WPM")
ax4.legend(title="Participant", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
fig4.savefig(target_dir / "trends.png", dpi=300)
plt.close(fig4)

print("Generated 4 visualization files successfully.")
