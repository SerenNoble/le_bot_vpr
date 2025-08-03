# Le-Bot VPR

## Preparation

```bash
# Clone the repository
git clone
# Update the submodules
git submodule update --init --recursive
# Navigate to the project directory
cd le-bot-vpr
# Create a virtual environment (optional but recommended)
python -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install the required packages
pip install -r requirements.txt
# Install dependencies for the VPR
pip install -r deps/vpr/requirements.txt
```

## Usage

```bash
# Develop server
python -m uvicorn app.main:app --reload
```