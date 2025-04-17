# 🚀 Git Workflow for Team Collaboration

This document outlines the step-by-step Git workflow to contribute a new feature or update to the team’s project using the `dev_branch`.

---

## 📦 1. Clone the Repository and Checkout `dev_branch`

```bash
git clone https://github.com/Trihamhochoi/python_written_assignment.git
cd python_written_assignment
git checkout dev_branch
```

> 🔍 You now have the `dev_branch` branch locally and are ready to begin working.

---

## 🌱 2. Create a New Feature Branch

Create a dedicated branch for your task or feature:

```bash
git checkout -b feat_branch/front_end
```

✅ This keeps your changes isolated and reviewable.

---

## ✍️ 3. Make Changes Locally

Edit or add your Python files (e.g., `model.py`, `dataloader.py`, etc.). Once ready, stage and commit your changes:

```bash
git add .
git commit -m "feat: Integrate the front-end code to model"
```

> 📝 Use clear, concise commit messages describing your changes.

---

## ☁️ 4. Push Your Feature Branch to Remote

```bash
git push origin feat_branch/front_end
```

> 📡 Your code is now available on GitHub for your team to review.

---

## 🔁 5. Open a Pull Request (PR)

1. Go to the GitHub repository in your browser.
2. Click the **"Pull requests"** tab. 
3. Select the green **"New Pull requests"** button.
4. Set the base branch to `dev_branch`.
5. Add a meaningful PR title and description.
6. Submit the PR.

---

## 👥 6. Peer Review and Merge

- One or more teammates will review code.
- Once approved, it will be **merged into** `dev_branch`.

> 🎉 Your feature is now integrated with the rest of the team’s work.

---

## ✅ Full Git Command Summary

```bash
git clone https://github.com/Trihamhochoi/python_written_assignment.git
cd python_written_assignment
git checkout dev_branch

# Create and switch to a new branch
git checkout -b feat_branch/front_end

# Make changes
git add .
git commit -m "feat: Integrate the front-end code to model"
git push origin feat_branch/front_end
```

After pushing, open a Pull Request on the GitHub UI to get your code reviewed and merged.

---

