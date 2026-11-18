# Self Tracked Analysis and Prediction of Typing Performance Under Variable Conditions

An exploratory data analysis (EDA) and machine learning mini-project investigating how physiological and lifestyle factors - **sleep duration**, **sleep quality**, **psychological stress**, and **caffeine dosage** - affect daily typing speed (WPM) and accuracy (%). 

The predictive pipeline is designed to be **strictly anonymous**: instead of relying on personal identifiers (`person_id`), user capability is abstracted into historical baseline averages (`baseline_wpm` and `baseline_acc`).

---

## Project Structure

```text
├── data/
│   ├── raw_response_form.csv           
│   └── cleaned_form_responses.csv  
├── plots/
│   ├── histograms.png             
│   ├── boxplots.png               
│   ├── pearson_scatter.png       
│   └── trends.png                
├── src/
│   ├── clean.py               
│   ├── visualize.py                
│   ├── train.py                         
│   └── predict.py                       
├── typing_model.joblib                  
├── requirements.txt    
└── README.md
```

---

## Dataset Overview (`n = 100`)

Data was collected across **5 participants (`P1` - `P5`)** spanning October - November 2026 via daily self-tracking protocols.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `sleep_hours` | Numeric (Float) | Total sleep duration preceding the test session (`4.0`  -  `10.0` hrs) |
| `sleep_quality` | Discrete (Int) | Subjective restfulness rating (`1` = poor to `5` = excellent) |
| `stress` | Categorical | Pre-test psychological stress state (`low`, `medium`, `high`) |
| `caffeine_mg` | Numeric (Float) | Caffeine consumed prior to testing (`0`, `50`, `100` mg) |
| `baseline_wpm` / `acc` | Derived (Float) | Anonymous historical average speed & accuracy per user |
| `wpm_avg` *(Target 1)* | Numeric (Float) | Mean typing speed across two consecutive benchmark runs |
| `acc_avg` *(Target 2)* | Numeric (Float) | Mean keystroke accuracy (%) across two consecutive benchmark runs |

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/abhishek9sm/EDS-
cd EDS-/
```

### 2. Install Dependencies:
```
pip install -r requirements.txt
```

### 3. Run the Pipeline

* **Step 1 — Clean Raw Form Data:**
  ```bash
  python3 clean.py
  ```
* **Step 2 — Generate EDA Visualizations:**
  ```bash
  python3 visualize.py
  ```
* **Step 3 — Train & Evaluate the ML Model:**
  ```bash
  python3 train.py
  ```
* **Step 4 — Run Interactive Predictions:**
  ```bash
  python3 predict.py
  ```

---

## Key Results & Findings

* **Model Performance (80/20 Holdout Split):**

  * **Typing Speed (`wpm_avg`):** $R^2 = 0.986$, $\text{MAE} = 2.75\text{ WPM}$
  * **Typing Accuracy (`acc_avg`):** $\text{MAE} = 3.23\%$

* **Dominant Variance Factor:** Inherent between-subject typing skill accounts for $>90\%$ of raw speed variance, validating the use of anonymous baseline normalization.

* **Simpson's Paradox in Caffeine:** Aggregate cohort correlation between caffeine and speed appears negative ($r = -0.35$) because the fastest typist (`P1`, ~105 WPM) consumes `0 mg` of caffeine; within-subject analysis confirms acute speed gains under moderate intake.

* **Sleep Stabilization:** Sessions following $\ge 7$ hours of sleep exhibit stable motor accuracy, whereas acute sleep restriction ($< 5$ hours) triggers sharp accuracy drops.

---

## Team & Academic Credits

* **Team Members:**
  * Ayush Shetty *NN25ISE055*
  * Abhishek S M *NN25ISE008*
  * Arun Ashok Shejekant *NN25ISE042*
* **Project Mentor:** *Dr. Vandana B S / Associate Professor / Dept of Information Science and Engineering/*
* **Institution:** *NMAM Institute of Technology, Nitte - 574110*
* **Course:** B.Tech in Information Science and Engineering
