{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "501ca8aa-91d2-4fcf-8bd9-ce6f19219925",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Risk tiers and threshold, loaded from the saved model metadata.\"\"\"\n",
    "import json\n",
    "\n",
    "def load_config(path: str = \"models/model_config.json\") -> dict:\n",
    "    with open(path) as f:\n",
    "        return json.load(f)\n",
    "\n",
    "def risk_tier(score: float, cfg: dict) -> str:\n",
    "    if score >= cfg[\"high_min\"]:\n",
    "        return \"High\"\n",
    "    if score >= cfg[\"low_max\"]:\n",
    "        return \"Medium\"\n",
    "    return \"Low\"\n",
    "\n",
    "def predict_label(score: float, cfg: dict) -> str:\n",
    "    return \"Yes\" if score >= cfg[\"threshold\"] else \"No\""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
