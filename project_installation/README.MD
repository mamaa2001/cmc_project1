# 🚀 Installation Guide

## Requirements

* **Python ≥ 3.10**
* Windows users: **Microsoft C++ Build Tools** (see below)

---

## 1. Install C++ Build Tools (Windows Only)

If you do not have these installed already, follow these instructions:

[https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

During installation, select:

*  MSVC v143 (or latest)
*  Windows 10/11 SDK

---

##  2. Activate your python virtual environment (for example if this is called venv_cmc)

**Linux / macOS**

```bash
source cmc_env/bin/activate
```

**Windows**

```bash
cmc_env\Scripts\activate
```

---

## 3. Install Project Dependencies

Make sure your environment is activated before continuing.

---

### Step 1 — Install the requirements

Install the requirements for the project by running the following commands:

```bash
pip install uv
uv pip install -r project_requirements.txt
uv pip install dm_control==1.0.38 --no-deps
```

---

### Step 2 — Install FARMS Packages

```bash
git submodule update --init --recursive
cd cmc_project_pack/farms
python setup_farms.py
```

---

Now you are ready to run the example, please refer to project_installation.pdf for
detailed instructions.
