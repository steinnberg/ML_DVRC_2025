# ⚙️ DVRC Lab 2 — Building your Kaggle, GitHub, and HuggingFace Profiles

## 🎯 Objective
By the end of this lab, you will:
- Create and customize your **Kaggle**, **GitHub**, and **HuggingFace** profiles  
- Connect them to **VSCode** for seamless workflow  
- Be ready to share datasets, notebooks, and AI models

---

## 🧩 Step 1. Create your Profiles

### 🏅 1. Kaggle
🔗 https://www.kaggle.com  
**Tasks:**
- Create your account using your academic email.  
- Add a **profile photo** and a **short bio**.  
- Download your **Kaggle API token**:
  - Go to `Account > API > Create New Token`.
  - It will download a file `kaggle.json`.

**Install Kaggle API locally:**
    ```bash
    pip install kaggle
    mkdir ~/.kaggle
    mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

---

### 🐙 2. GitHub

🔗 https://github.com

**Tasks:**
- Create a GitHub account (use your real name and a clean profile picture).
- Create your first repository:
```bash
    echo "# DVRC-Lab" >> README.md
    git init
    git add README.md
    git commit -m "first commit"
    git branch -M main
    git remote add origin https://github.com/<username>/DVRC-Lab.git
    git push -u origin main
```   

**Connect GitHub to VSCode:**
- Open VSCode → Source Control tab
- Click “Publish to GitHub”
- Log in when prompted

>You’re ready to pull, push, commit directly from VSCode

### 🤗 3. HuggingFace

🔗 https://huggingface.co

**Tasks:**
- Create an account and add a short description (e.g. “DVRC Student - Machine Learning Explorer”)
- Create your first model or dataset repository
- Generate an access token:
    * Profile → Settings → Access Tokens → New Token

>Copy it safely.

#### Install HuggingFace CLI:
```Python
    pip install huggingface_hub
    huggingface-cli login
```

--- 

## 🧠 Step 2. Connect Everything to VSCode

**Extensions to install:**
    - GitHub Pull Requests and Issues
    - Kaggle (for code snippets)
    - HuggingFace (optional)
    - Python & Jupyter extensions

Then, open your project folder:

DVRC/
│
├── data/
├── notebooks/
├── src/
├── requirements.txt
└── README.md


**Login to each service inside VSCode terminal:** 

kaggle datasets list
```bash
    git config --global user.name "Your Name"
    git config --global user.email "your@email.com"
    huggingface-cli whoami
```

---

## 🚀 Step 3. Deliverables
Platform	Deliverable
    - Kaggle	Screenshot of your profile + one downloaded dataset
    - GitHub	Repository “DVRC-Lab” with your README.md
    - HuggingFace	Created profile + token setup

---


## 🧭 Step 4. Optional Bonus

 - Publish a simple notebook on Kaggle with your cleaned dataset from Lab 1.
 - Sync your VSCode project with GitHub using push/pull.
 - Upload a simple trained model to HuggingFace Hub.

## 🎓 Evaluation Criteria
Criterion	Weight
Profile setup (3 platforms)	40%
Connection to VSCode	30%
Clean structure and screenshots	20%
Bonus (public notebook/model)	10%