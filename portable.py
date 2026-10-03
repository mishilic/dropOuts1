import os
import urllib.request
import shutil
import numpy as np

# 1. Download portable.py from the reference repository
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/aaubs/tonights-front-desk/main/portable.py", 
    "portable.py"
)
import portable

# 2. Create the model directory
os.makedirs("model", exist_ok=True)

# 3. Export your trained XGBoost model (assuming your model object is named 'booster')
portable.export(booster, "model")

# 4. Verify the exported model works correctly
back = portable.Model("model").predict_proba(val)
is_valid = np.allclose(back, booster.predict_proba(val[features])[:, 1], atol=1e-5)
print(f"Model exported successfully! Predictions match: {is_valid}")

# 5. Create a zip file of the model folder for easy download (if using Colab)
shutil.make_archive("model", "zip", "model")
print("Downloaded model archived to model.zip")
