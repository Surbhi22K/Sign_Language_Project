You are a senior ML engineer and full-stack developer. Refactor and build a complete end-to-end sign language decoding system from the existing monolithic codebase.

Follow ALL instructions strictly and implement in a modular, production-quality way.

---

# 🎯 OBJECTIVE

Build a system that:

1. Trains a sign language model (frame-level classification)
2. Performs video-based inference (frame-by-frame)
3. Decodes predictions into final words/sentences (handle redundancy)
4. Computes research-grade evaluation metrics
5. Generates a LaTeX research paper
6. Deploys model to a mobile app (Expo) for real-time inference

---

# 📁 STEP 1 — RESTRUCTURE PROJECT

Create the following structure:

project_root/

* new_code/

  * data/

    * dataset.py
    * transforms.py
  * models/

    * model.py
    * layers.py
  * training/

    * train.py
    * loss.py
    * optimizer.py
  * evaluation/

    * metrics.py
    * evaluate.py
  * inference/

    * frame_infer.py
    * sequence_decoder.py
  * utils/

    * config.py
    * logger.py
    * checkpoint.py

* video/

* test/

  * test_video.py

* saved_models/

* reports/

  * paper.tex

* expo_app/

* main.py

Refactor existing code into these modules. Do NOT leave logic in a single file.

---

# 🧠 STEP 2 — DATA PIPELINE

Implement:

* Video loader:

  * Read video from `video/`
  * Extract frames
  * Resize + normalize
* Dataset class:

  * Returns (sequence, label)
  * Sequence shape: (T, C, H, W)

---

# 🤖 STEP 3 — MODEL

* Move existing architecture into `models/model.py`
* Ensure:

  * Input: sequence
  * Output: per-frame logits (T, num_classes)

---

# 🏋️ STEP 4 — TRAINING

Implement in `training/train.py`:

* Training loop
* Validation loop
* Loss + optimizer
* Early stopping
* Save checkpoints to:
  saved_models/sign_model.pt

---

# 📊 STEP 5 — METRICS

In `evaluation/metrics.py`, implement:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Sequence-level:

* Sequence accuracy
* Word accuracy rate
* Edit distance (Levenshtein)

Also:

* Save results to reports/results.json
* Generate plots (loss curve, confusion matrix)

---

# 🔁 STEP 6 — SEQUENCE DECODING (CRITICAL)

In `sequence_decoder.py`, implement:

* Majority voting
* Sliding window smoothing
* Collapse repeats:
  Example: A A A B B C → A B C

Return final word/sentence

---

# 🎥 STEP 7 — INFERENCE

In `test/test_video.py`:

* Load model from saved_models/
* Load video (hardcoded filename)
* Extract frames
* Run frame-by-frame inference
* Store predictions
* Decode sequence
* Print final output

---

# 📄 STEP 8 — RESEARCH PAPER

Create `reports/paper.tex` with:

Sections:

* Introduction
* Motivation
* Related Work
* Methodology
* Experiments
* Results
* Discussion
* Conclusion

Include:

* Metrics table
* Graphs

---

# 📱 STEP 9 — MOBILE APP

Inside `expo_app/`:

* Initialize Expo app
* Install camera library

---

# 🔄 STEP 10 — MODEL CONVERSION

Convert model:
PyTorch → ONNX → TensorFlow Lite

Prepare model for mobile inference.

---

# 📷 STEP 11 — MOBILE INFERENCE

Implement:

* Camera input (real-time)
* Frame capture every 200–300 ms
* Preprocessing
* Model inference
* Prediction buffer
* Sequence decoding (same logic as backend)

---

# 🎨 STEP 12 — UI

Display:

* Camera preview
* Large, real-time predicted text

Keep UI simple and clean.

---

# ⚡ STEP 13 — OPTIMIZATION

* Use frame skipping
* Use lightweight model
* Keep model size small (<20MB)

---

# 🚀 EXECUTION ORDER (MANDATORY)

Execute strictly in this order:

1. Refactor project
2. Implement dataset
3. Implement model
4. Implement training
5. Train and save model
6. Implement inference
7. Implement sequence decoder
8. Build test pipeline
9. Add metrics
10. Generate results
11. Create LaTeX paper
12. Convert model
13. Build Expo app
14. Integrate inference
15. Optimize

---

# ⚠️ REQUIREMENTS

* Code must be clean, modular, and reusable
* Avoid duplication
* Add comments where necessary
* Ensure everything runs end-to-end

---

# ✅ FINAL OUTPUT

System should:

* Train successfully
* Predict from video files
* Output decoded words/sentences
* Generate evaluation metrics
* Produce a research paper (LaTeX)
* Run on mobile with real-time predictions

---

