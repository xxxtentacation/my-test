---
name: latex-paper-format
description: Applies a specific academic LaTeX paper template and formatting structure. Use when generating, writing, or editing LaTeX manuscripts in this workspace to ensure consistent styling, packages, margin setups, and macro definitions.
---

# Academic LaTeX Paper Format

## Overview

This skill ensures that whenever you create or edit LaTeX documents in this workspace, you follow the established template structure. It standardizes the document class, page geometry, package imports, theorem environments, and citation styles required for the paper.

## When to Use

- Creating a new `.tex` manuscript.
- Formatting or restructuring an existing LaTeX paper.
- Adding new sections, theorems, elements, or citations.
- Regenerating the preamble or abstract sections of the document.

## Core Formatting Rules

### 1. Document Class and Margins

Always use the standard `12pt` article class and manual margin resets (do not use the `geometry` package unless explicitly requested):

```latex
\documentclass[12pt]{article}
\textheight=24.5cm \textwidth=16.5cm
\parskip = 0.165cm
\renewcommand\baselinestretch{1.2}
\topmargin=-2cm \oddsidemargin=0cm \evensidemargin=0cm
```

### 2. Required Packages

Include the following standardized packages:
- **Math & Symbols:** `amsfonts`, `amsmath`, `amssymb`, `amscd`, `latexsym`
- **Citation:** `\usepackage[round, sort]{natbib}`
- **Hyperlinks (Black Links):** `\usepackage[colorlinks=true,linkcolor=black,anchorcolor=black,citecolor=black,urlcolor=black]{hyperref}`
- **Colors & Tables:** `color`, `xcolor`, `array`, `colortbl`
- **Algorithms:** `algorithmicx`, `algorithm`, `algpseudocode`
- **Lists & Floats:** `enumerate`, `float`

### 3. Theorem Environments

Define the following theorem-like environments exactly as shown, numbered by section:

```latex
\newtheorem{theorem}{Theorem}[section]
\newtheorem{cor}{Corollary}[section]
\newtheorem{lemma}{Lemma}[section]
\newtheorem{rem}{Remark}[section]
\newtheorem{definition}{Definition}[section]
\newtheorem{obs}{Observation}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{exam}{Example}[section]
```

### 4. Title, Authors, and Abstract Structure

Ensure the title uses `\LARGE\bf` and authors include footnote affiliations for the corresponding author. Suppress the auto-date with `\date{}`. The document body must be enclosed in `\begin{sloppypar}` to prevent text overflow.

```latex
\title{\LARGE\bf Paper Title Here}
\author{Author One$^{a,}$\footnote{Corresponding author. E-mail address: author@email.com}, Author Two$^{b}$\\
{\small $^{a}$Institution One, Location, Country}\\
{\small $^{b}$Institution Two, Location, Country}\\}
\date{}

\begin{document}
\begin{sloppypar}

\maketitle 

\begin{abstract}	
Abstract text goes here...
\end{abstract}

\noindent \textbf{Keywords:} Keyword 1; Keyword 2; Keyword 3
```

### 5. Bibliography and Citations

The document relies on `natbib` (round and sorted). Use `apalike` as the bibliography style and reference the local `.bib` file.

```latex
\bibliographystyle{apalike}
\bibliography{ref_IJPR} % Or the specified .bib filename
```

## Best Practices

- Always use `\citealp{...}` or `\cite{...}` consistent with the existing text and `natbib` format.
- Ensure all environments like equations, theorems, algorithms, and tables are properly aligned with `\baselinestretch{1.2}` linespacing.
- Always close the document with `\end{sloppypar}` before `\end{document}`.