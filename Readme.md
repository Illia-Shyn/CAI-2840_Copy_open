**Phase 1: Discover & Frame**

---

## 1. README.md

**Problem Statement**
Local youth soccer clubs in Miami record match footage on smartphones but lack affordable tools to extract tactical insights. Coaches need a lightweight, AI‑driven solution to automatically analyze video, generate statistics (possession %, shot count, heat‑maps), and compile highlight reels without the expense of enterprise platforms.

**Stakeholder Map**

* **Beneficiaries:**

  * Youth coaches & teams (tactical insights, training improvements)
  * Players (performance feedback, recruitment exposure)
  * Parents & fans (engaging highlight reels)
* **Implementers:**

  * Project lead (Illia Shybanov)
  * Data annotators (hand‑label frames)
  * ML engineers (detection & tracking models)
  * Front‑end developer (dashboard & export tools)

**SMART Goals**

1. **S**: Achieve ≥70% mAP\@0.5 for player detection and ≥60% for ball within 6 weeks.
2. **M**: Generate highlight reels of top 5 shots per match in <3 minutes on Colab.
3. **A**: Leverage YOLOv8‑n and BoT‑SORT with public SoccerNet & Roboflow datasets.
4. **R**: Use free Google Colab resources to ensure zero infrastructure cost.
5. **T**: Deliver the interactive dashboard and demo video by Week 6 (deadline: Aug 29, 2025).

**Data Sources & Model Types**

* **Video data:** Sample matches from SoccerNet (broadcast clips) + user‑recorded phone videos.
* **Annotations:** Roboflow “soccer‑player” and “soccer‑ball” collections; 200–300 hand‑labels.
* **Model types:**

  * **Detection:** Ultralytics YOLOv8‑n (fine‑tune on soccer classes)
  * **Tracking:** BoT‑SORT (integrated via `model.track(tracker="botsort.yaml")`)

---

## 2. GitHub Repository Structure

```
/ai‑soccer‑analyzer/
├── README.md
├── data/
│   ├── raw/           # Original video clips
│   └── processed/     # Cropped & annotated frames
├── notebooks/         # EDA & model‑testing notebooks
├── src/
│   ├── detection/     # Training & inference scripts
│   ├── tracking/      # BoT‑SORT integration
│   └── analytics/     # Stats computation & utilities
├── slides/            # PowerPoint templates & exports
└── planner/           # Microsoft Planner board export
```

---

