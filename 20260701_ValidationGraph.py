"""
Validation Report Graphs
========================
Paste this as a new cell in Colab, AFTER your simulation cell.
Paste the CSV data directly — no file needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ── Raw data (paste your CSV here) ──────────────────────────────────────────
CSV_DATA = """Scenario,Metric,Fixed Mean,Fixed Std,Fixed CI95,Adaptive Mean,Adaptive Std,Adaptive CI95,Improvement %,p-value
VS-01,Travel Time,25.888888888888925,6.861403075984938,2.4553215573568252,21.556944444444504,5.748552659478712,2.0570931502071654,16.732832618025636,0.010412195919843322
VS-01,Waiting Time,11.775000000000018,3.9023975830958015,1.3964550406152714,7.50055555555556,3.498391073360902,1.2518831678249247,36.301014390186424,3.7813824842774966e-05
VS-01,Collisions,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
VS-02,Travel Time,31.643717948717548,4.36304787168305,1.5612966293465318,30.108974358973967,2.9855415434771118,1.0683623204910537,4.850073535072009,0.11797318546364158
VS-02,Waiting Time,17.260448717948766,3.2238352663496643,1.1536346340794525,15.506410256410298,2.2453439397125723,0.803486009756889,10.162183441468013,0.017906611876925223
VS-02,Collisions,0.06666666666666667,0.36514837167011066,0.13066666666666665,0.0,0.0,0.0,100.0,0.32558198801619376
VS-03,Travel Time,28.034062500000076,3.0080641380407025,1.0764219274471214,27.257812500000068,3.8325582396475584,1.3714633525274085,2.768952947864784,0.38664414491027993
VS-03,Waiting Time,14.500520833333358,2.3287711125154438,0.8333400401328585,12.738229166666692,2.733181905142525,0.9780565836981839,12.153299091268249,0.009420553701624328
VS-03,Collisions,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
VS-11,Travel Time,35.289084967318985,3.0489118349073316,1.0910390880444987,34.170225225223966,5.391039031312842,1.9291585414160175,3.170554700217328,0.32762978084298344
VS-11,Waiting Time,20.623300653594836,2.6429430688701707,0.9457650308544533,19.785200494612333,4.312961079731771,1.5433733010338067,4.063850753377897,0.36866555974811765
VS-11,Collisions,0.16666666666666666,0.461133037377414,0.16501433816722194,0.0,0.0,0.0,100.0,0.0573072855932361
VS-12,Travel Time,33.27645021644928,3.3464905564254543,1.1975262659383943,33.224805194804176,4.537219183518052,1.6236230328365657,0.15519991257832927,0.9601714137127818
VS-12,Waiting Time,18.847878787878848,2.6204616993094536,0.9377201760762036,18.58086580086587,3.766043865887976,1.3476614895618593,1.4166739399061519,0.7511892989735307
VS-12,Collisions,0.13333333333333333,0.43417248545530474,0.15536662856620997,0.1,0.30512857662936466,0.10918885884810649,24.999999999999993,0.73219866449991
VS-13,Travel Time,30.34885416666668,3.7046022245755945,1.3256748805894631,26.08968750000006,5.3320614850884835,1.9080536975464728,14.034027918406972,0.0007282452885742164
VS-13,Waiting Time,15.460312500000036,2.200987537650877,0.7876132751306358,11.810416666666688,3.4553395178403283,1.2364773665366458,23.60816337530913,1.158014959480343e-05
VS-13,Collisions,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
VS-14,Travel Time,29.600060606060243,2.996449441939168,1.0722656618255813,29.52587878787836,4.843530237932094,1.733235035925531,0.2506137374823364,0.9434220624408172
VS-14,Waiting Time,15.70630303030307,2.235866077806856,0.8000944004319385,15.122969696969744,3.5902263541533666,1.2847460010032956,3.714007887202153,0.4536482098689457
VS-14,Collisions,0.06666666666666667,0.36514837167011066,0.13066666666666665,0.0,0.0,0.0,100.0,0.32558198801619376
"""

# ── Parse CSV ────────────────────────────────────────────────────────────────
import csv

rows = list(csv.DictReader(io.StringIO(CSV_DATA.strip())))

SCENARIOS = ['VS-01', 'VS-02', 'VS-03', 'VS-11', 'VS-12', 'VS-13', 'VS-14']
METRICS   = ['Travel Time', 'Waiting Time', 'Collisions']

data = {}  # data[scenario][metric] = dict of values
for row in rows:
    s, m = row['Scenario'], row['Metric']
    if s not in data:
        data[s] = {}
    data[s][m] = {
        'fixed_mean':  float(row['Fixed Mean']),
        'fixed_ci':    float(row['Fixed CI95']),
        'adap_mean':   float(row['Adaptive Mean']),
        'adap_ci':     float(row['Adaptive CI95']),
        'improvement': float(row['Improvement %']),
        'pval':        float(row['p-value']) if row['p-value'] else None,
    }

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#F8F9FA',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.size':        10,
})

# הגדרת הצבעים החדשה לבקשתך
C_FIXED = '#2B6CB0'   # כחול עמוק ומקצועי
C_ADAP  = '#C53030'   # אדום ברור ומנוגד
C_SIG   = '#2D3748'   # אפור כהה מאוד לסימוני מובהקות משמעותיים
C_NSIG  = '#A0AEC0'   # אפור בהיר לסימונים לא משמעותיים


def sig_label(pval):
    """Return a significance annotation string."""
    if pval is None:
        return ''
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return 'ns'


x      = np.arange(len(SCENARIOS))
width  = 0.35

# ── Figure 1: Waiting Time ────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(12, 5))

fixed_means = [data[s]['Waiting Time']['fixed_mean'] for s in SCENARIOS]
fixed_cis   = [data[s]['Waiting Time']['fixed_ci']   for s in SCENARIOS]
adap_means  = [data[s]['Waiting Time']['adap_mean']  for s in SCENARIOS]
adap_cis    = [data[s]['Waiting Time']['adap_ci']    for s in SCENARIOS]
pvals       = [data[s]['Waiting Time']['pval']       for s in SCENARIOS]

b1 = ax.bar(x - width/2, fixed_means, width, label='Fixed',    color=C_FIXED, alpha=0.9)
b2 = ax.bar(x + width/2, adap_means,  width, label='Adaptive', color=C_ADAP,  alpha=0.9)
ax.errorbar(x - width/2, fixed_means, yerr=fixed_cis, fmt='none',
            color='#1A202C', capsize=5, linewidth=1.5)
ax.errorbar(x + width/2, adap_means,  yerr=adap_cis,  fmt='none',
            color='#1A202C', capsize=5, linewidth=1.5)

# Significance annotations
top = max(max(fixed_means), max(adap_means))
for i, pval in enumerate(pvals):
    lbl = sig_label(pval)
    col = C_SIG if (pval is not None and pval < 0.05) else C_NSIG
    ax.text(x[i], top * 1.04, lbl, ha='center', va='bottom',
            fontsize=12, color=col, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SCENARIOS)
ax.set_ylabel('Avg Waiting Time (s)')
ax.set_title('Average Waiting Time: Fixed vs Adaptive (30 runs, 95% CI)\n'
             '*** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
             fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, top * 1.18)
plt.tight_layout()
plt.show()

# ── Figure 2: Travel Time ─────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 5))

fixed_means = [data[s]['Travel Time']['fixed_mean'] for s in SCENARIOS]
fixed_cis   = [data[s]['Travel Time']['fixed_ci']   for s in SCENARIOS]
adap_means  = [data[s]['Travel Time']['adap_mean']  for s in SCENARIOS]
adap_cis    = [data[s]['Travel Time']['adap_ci']    for s in SCENARIOS]
pvals       = [data[s]['Travel Time']['pval']       for s in SCENARIOS]

b1 = ax.bar(x - width/2, fixed_means, width, label='Fixed',    color=C_FIXED, alpha=0.9)
b2 = ax.bar(x + width/2, adap_means,  width, label='Adaptive', color=C_ADAP,  alpha=0.9)
ax.errorbar(x - width/2, fixed_means, yerr=fixed_cis, fmt='none',
            color='#1A202C', capsize=5, linewidth=1.5)
ax.errorbar(x + width/2, adap_means,  yerr=adap_cis,  fmt='none',
            color='#1A202C', capsize=5, linewidth=1.5)

top = max(max(fixed_means), max(adap_means))
for i, pval in enumerate(pvals):
    lbl = sig_label(pval)
    col = C_SIG if (pval is not None and pval < 0.05) else C_NSIG
    ax.text(x[i], top * 1.04, lbl, ha='center', va='bottom',
            fontsize=12, color=col, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SCENARIOS)
ax.set_ylabel('Avg Travel Time (s)')
ax.set_title('Average Travel Time: Fixed vs Adaptive (30 runs, 95% CI)\n'
             '*** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
             fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, top * 1.18)
plt.tight_layout()
plt.show()

# ── Figure 3: % Improvement (Waiting Time) ────────────────────────────────────
fig3, ax = plt.subplots(figsize=(10, 5))

improvements = [data[s]['Waiting Time']['improvement'] for s in SCENARIOS]
pvals        = [data[s]['Waiting Time']['pval']        for s in SCENARIOS]

# צביעת הברים באדום (Adaptive) או אפור בהתאם למובהקות
bar_colors   = [C_ADAP if (p is not None and p < 0.05) else '#CBD5E0' for p in pvals]

bars = ax.barh(SCENARIOS, improvements, color=bar_colors, alpha=0.85,
               edgecolor='black', linewidth=0.5)
ax.axvline(0,  color='black', linewidth=1.0)
ax.axvline(15, color='orange', linewidth=2, linestyle='--',
           label='Acceptance threshold (15%)')

for bar, pval, imp in zip(bars, pvals, improvements):
    lbl = sig_label(pval)
    ax.text(imp + 0.5, bar.get_y() + bar.get_height()/2,
            f'{imp:.1f}% {lbl}', va='center', fontsize=9)

ax.set_xlabel('Wait Time Reduction — Adaptive vs Fixed (%)')
ax.set_title('Waiting Time Improvement: Adaptive over Fixed\n'
             '(positive = adaptive better; *** p<0.001  ** p<0.01  * p<0.05)',
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.show()

# ── Figure 4: p-value Heatmap ─────────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(9, 4))

pval_matrix = []
for metric in ['Waiting Time', 'Travel Time', 'Collisions']:
    row = []
    for s in SCENARIOS:
        pv = data[s][metric]['pval']
        row.append(pv if pv is not None else 1.0)
    pval_matrix.append(row)

pval_matrix = np.array(pval_matrix)

im = ax.imshow(pval_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
plt.colorbar(im, ax=ax, label='p-value')

ax.set_xticks(range(len(SCENARIOS)))
ax.set_yticks(range(3))
ax.set_xticklabels(SCENARIOS)
ax.set_yticklabels(['Waiting Time', 'Travel Time', 'Collisions'])

for i in range(3):
    for j in range(len(SCENARIOS)):
        pv = pval_matrix[i, j]
        lbl = sig_label(pv) if pv < 1.0 else 'N/A'
        color = 'white' if pv < 0.05 else 'black'
        ax.text(j, i, f'{pv:.3f}\n{lbl}', ha='center', va='center',
                fontsize=8, color=color)

ax.set_title('Statistical Significance Heatmap (p-values)\n'
             'Green = significant (p<0.05), Red = not significant',
             fontsize=11)
plt.tight_layout()
plt.show()

# ── Figure 5: Collision Rate ───────────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(12, 4))

fixed_col  = [data[s]['Collisions']['fixed_mean'] for s in SCENARIOS]
fixed_ci_c = [data[s]['Collisions']['fixed_ci']   for s in SCENARIOS]
adap_col   = [data[s]['Collisions']['adap_mean']  for s in SCENARIOS]
adap_ci_c  = [data[s]['Collisions']['adap_ci']    for s in SCENARIOS]

# עדכון משתני הצבעים לצבעים שהגדרת בתחילת הסקריפט
b1 = ax.bar(x - width/2, fixed_col, width, label='Fixed',    color=C_FIXED, alpha=0.85)
b2 = ax.bar(x + width/2, adap_col,  width, label='Adaptive', color=C_ADAP,  alpha=0.85)
ax.errorbar(x - width/2, fixed_col, yerr=fixed_ci_c, fmt='none',
            color='black', capsize=5, linewidth=1.5)
ax.errorbar(x + width/2, adap_col,  yerr=adap_ci_c,  fmt='none',
            color='black', capsize=5, linewidth=1.5)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(SCENARIOS)
ax.set_ylabel('Avg Collision Count per Run')
ax.set_title('Collision Rate: Fixed vs Adaptive (30 runs, 95% CI)', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
