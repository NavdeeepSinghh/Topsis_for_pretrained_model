# GitHub Upload Guide - Step by Step

## Prerequisites
- A GitHub account (create one at https://github.com if you don't have one)
- Git installed on your computer (download from https://git-scm.com)

## Method 1: Using GitHub Web Interface (Easiest)

### Step 1: Create a New Repository
1. Go to https://github.com and log in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Enter repository name: `topsis-nlp-model-selection`
5. Add description: "TOPSIS analysis for selecting best pre-trained NLP models"
6. Choose **Public** or **Private**
7. ✅ Check "Add a README file" (you'll replace it later)
8. Click **"Create repository"**

### Step 2: Upload Files
1. In your new repository, click **"Add file"** → **"Upload files"**
2. Drag and drop ALL files from your project folder:
   - `topsis.py`
   - `main.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - All files in `results/` folder
   - All files in `visualizations/` folder
3. Add commit message: "Initial commit - TOPSIS NLP assignment"
4. Click **"Commit changes"**

### Step 3: Verify Upload
1. Check that all files appear in your repository
2. Click on `README.md` to verify it displays properly
3. Check that images in `visualizations/` folder are visible

---

## Method 2: Using Git Command Line (Recommended)

### Step 1: Create Repository on GitHub
1. Go to https://github.com and log in
2. Click **"+"** → **"New repository"**
3. Name: `topsis-nlp-model-selection`
4. Description: "TOPSIS analysis for selecting best pre-trained NLP models"
5. Choose Public/Private
6. **DON'T** check "Add a README file"
7. Click **"Create repository"**

### Step 2: Initialize Git in Your Project Folder
Open terminal/command prompt in your project folder and run:

```bash
# Navigate to your project folder
cd path/to/topsis_nlp_assignment

# Initialize git repository
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit - TOPSIS NLP model selection assignment"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/topsis-nlp-model-selection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Enter Credentials
- Enter your GitHub username when prompted
- For password, use a **Personal Access Token** (not your GitHub password)
  - Generate token at: https://github.com/settings/tokens
  - Click "Generate new token (classic)"
  - Select scopes: `repo` (full control of private repositories)
  - Copy the token and use it as password

---

## Method 3: Using GitHub Desktop (User-Friendly)

### Step 1: Download GitHub Desktop
1. Download from https://desktop.github.com
2. Install and log in with your GitHub account

### Step 2: Create Repository
1. Click **"File"** → **"New repository"**
2. Name: `topsis-nlp-model-selection`
3. Local path: Choose where to create the repository
4. Click **"Create repository"**

### Step 3: Add Your Files
1. Copy all your project files to the repository folder
2. GitHub Desktop will automatically detect the changes
3. Add commit message: "Initial commit - TOPSIS NLP assignment"
4. Click **"Commit to main"**
5. Click **"Publish repository"** to upload to GitHub

---

## After Upload - Final Checks

### ✅ Verification Checklist
- [ ] All Python files (`.py`) are uploaded
- [ ] `requirements.txt` is present
- [ ] `README.md` displays correctly with all sections
- [ ] All CSV files in `results/` folder are uploaded
- [ ] All PNG images in `visualizations/` folder are uploaded
- [ ] Images display in README.md
- [ ] Repository is public (if required) or private

### 📝 Update README if Needed
If images don't display in README:
1. Click on each image in `visualizations/` folder
2. Click "Download" and note the GitHub URL
3. Edit README.md and update image paths to use full GitHub URLs

---

## Common Issues and Solutions

### Issue 1: Images Not Showing in README
**Solution:** Use relative paths like:
```markdown
![Image](visualizations/summarization_analysis.png)
```

Or use full GitHub URLs:
```markdown
![Image](https://github.com/YOUR_USERNAME/topsis-nlp-model-selection/blob/main/visualizations/summarization_analysis.png?raw=true)
```

### Issue 2: File Too Large Error
**Solution:** GitHub has a 100MB file size limit. If any file is too large:
- Reduce image resolution
- Compress images
- Use Git LFS for large files

### Issue 3: Authentication Failed
**Solution:** 
- Use Personal Access Token instead of password
- Generate at: https://github.com/settings/tokens
- Or use SSH keys

---

## Repository Structure

Your final repository should look like:

```
topsis-nlp-model-selection/
│
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Files to ignore
│
├── topsis.py                         # TOPSIS implementation
├── main.py                           # Main analysis script
│
├── results/                          # CSV results
│   ├── summarization_results.csv
│   ├── generation_results.csv
│   ├── classification_results.csv
│   ├── similarity_results.csv
│   ├── conversational_results.csv
│   └── overall_comparison.csv
│
└── visualizations/                   # Charts and graphs
    ├── summarization_analysis.png
    ├── summarization_radar.png
    ├── generation_analysis.png
    ├── generation_radar.png
    ├── classification_analysis.png
    ├── classification_radar.png
    ├── similarity_analysis.png
    ├── similarity_radar.png
    ├── conversational_analysis.png
    ├── conversational_radar.png
    └── overall_comparison.png
```

---

## Tips for a Professional Repository

1. **Add Topics/Tags**: In repository settings, add topics like:
   - `topsis`
   - `nlp`
   - `machine-learning`
   - `decision-making`
   - `python`

2. **Add License**: Consider adding an MIT or Apache 2.0 license

3. **Star Your Repo**: Star your own repository to bookmark it

4. **Share the Link**: Your repository URL will be:
   `https://github.com/YOUR_USERNAME/topsis-nlp-model-selection`

---

## Need Help?

If you encounter any issues:
1. Check GitHub's help documentation: https://docs.github.com
2. Search for error messages on Stack Overflow
3. Contact your instructor/TA

---

**Good luck with your assignment! 🚀**
