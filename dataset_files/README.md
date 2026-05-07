# Datasets

ICL task datasets, all in `.json` format.

* `abstractive/` — tasks that require information not present in the prompt
  (e.g. translation, country-capital). Inherited from Todd et al. (2024).
* `extractive/` — tasks where the answer is one of the in-context tokens
  (e.g. choose-first-of-list, animal vs. object). Inherited from Todd et al. (2024).
* `new_tasks/` — tasks introduced in this paper (capital-index, abstract-clf,
  binding tasks from Feng & Steinhardt (2024), capitalize-second-letter,
  french-english).
* `generate/` — scripts and notebook used to construct or filter the datasets.

Each task is provided in three JSON files: `<task>.json` (full),
`<task>_train.json`, and `<task>_val.json`.
