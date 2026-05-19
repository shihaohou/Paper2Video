r"""One-off patches to the generated slides_refined.tex for the FedPuReL deck.

- Replace frame 1 (title) with the short-affiliation version.
- Replace frame 15 (conclusion) with a tighter, "take-home"-style closer.
- Replace frame 16 (thank-you) with a clean Q&A + contact card.
- Hide default beamer navigation icons.
- Ensure \usepackage{hyperref} is loaded for clickable links.

Run from project root:

    python scripts/patch_slides.py src/result/mypaper/latex_proj/slides_refined.tex
"""
import argparse
import re
import sys
from pathlib import Path

NEW_FRAME_1 = r"""\begin{frame}
  \begin{columns}[T]
    \begin{column}{0.50\textwidth}
      \raggedright
      \includegraphics[height=0.95cm]{fig/logo1.png}
    \end{column}
    \begin{column}{0.48\textwidth}
      \raggedleft
      \includegraphics[height=1.5cm]{fig/cvpr_logo.png}
    \end{column}
  \end{columns}

  \vspace{0.7cm}
  \begin{center}
    {\Large \textbf{Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized Federated Learning}}
    \par
    \vspace{0.6cm}
    \small
    Shihao Hou\textsuperscript{1}, Chikai Shang\textsuperscript{1}, Zhiheng Yang\textsuperscript{1}, Jiacheng Yang\textsuperscript{1}, Xinyi Shang\textsuperscript{2}, \\
    Junlong Gao\textsuperscript{1}, Yiqun Zhang\textsuperscript{3}, Yang Lu\textsuperscript{1,*}
    \par
    \vspace{0.5cm}
    \footnotesize
    \textsuperscript{1}Xiamen University \quad
    \textsuperscript{2}University College London \quad
    \textsuperscript{3}Guangdong University of Technology
    \par
    \vspace{0.3cm}
    \scriptsize \textsuperscript{*}Corresponding author
  \end{center}
\end{frame}"""

NEW_FRAME_CONCLUSION = r"""\begin{frame}{Conclusion}
  \vspace{0.3cm}
  \begin{itemize}
    \item \alert{Foundation models carry an implicit balanced prior} -- preserve it during PEFT, don't overwrite it.
    \item \alert{Gradient Purification + Residual Learning}: a simple, prior-free recipe that beats prior-based methods.
    \item Consistent gains on ImageNet-LT, Places-LT, and CIFAR-100-LT across varying $\alpha$ and IF.
  \end{itemize}

  \vspace{0.6cm}
  \begin{block}{Take-home}
    \centering Reuse what foundation models already know -- don't fight it.
  \end{block}
\end{frame}"""

NEW_FRAME_THANKS = r"""\begin{frame}
  \vspace{2.2cm}
  \begin{center}
    {\Huge \textbf{Thank you!}}

    \vspace{1.2cm}
    \footnotesize
    \begin{tabular}{rl}
      Code:    & \href{https://github.com/shihaohou/FedPuReL}{github.com/shihaohou/FedPuReL} \\[2pt]
      Contact: & \href{mailto:houshihao@stu.xmu.edu.cn}{houshihao@stu.xmu.edu.cn} \quad
                 \href{mailto:luyang@xmu.edu.cn}{luyang@xmu.edu.cn} \\
    \end{tabular}
  \end{center}
\end{frame}"""

PREAMBLE_INJECTIONS = [
    r"\usepackage{hyperref}",
    r"\setbeamertemplate{navigation symbols}{}",
]


def ensure_preamble(tex: str) -> str:
    r"""Inject \usepackage{hyperref} and \setbeamertemplate{navigation symbols}{}
    just before \begin{document} if they're not already there.
    """
    doc_idx = tex.find(r"\begin{document}")
    if doc_idx == -1:
        raise RuntimeError(r"missing \begin{document}")
    preamble = tex[:doc_idx]
    rest = tex[doc_idx:]
    additions = []
    for line in PREAMBLE_INJECTIONS:
        if line not in preamble:
            additions.append(line)
    if additions:
        injection = "\n" + "\n".join(additions) + "\n"
        preamble = preamble.rstrip() + injection
    return preamble + rest


def replace_nth_frame(tex: str, n: int, new_frame: str) -> str:
    """Replace the n-th frame (1-indexed) with new_frame."""
    pattern = re.compile(r"\\begin\{frame\}.*?\\end\{frame\}", flags=re.DOTALL)
    matches = list(pattern.finditer(tex))
    if len(matches) < n:
        raise RuntimeError(f"found {len(matches)} frames, can't replace #{n}")
    m = matches[n - 1]
    return tex[: m.start()] + new_frame + tex[m.end():]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tex_path")
    args = p.parse_args()

    path = Path(args.tex_path)
    tex = path.read_text()

    pattern = re.compile(r"\\begin\{frame\}.*?\\end\{frame\}", flags=re.DOTALL)
    n_frames = len(pattern.findall(tex))
    print(f"found {n_frames} frames")
    if n_frames < 16:
        print(f"WARNING: expected 16 frames, got {n_frames} — frame numbering may be off",
              file=sys.stderr)

    tex = ensure_preamble(tex)
    tex = replace_nth_frame(tex, 1, NEW_FRAME_1)
    # Conclusion / Thanks are last two frames, even if total count drifted
    tex = replace_nth_frame(tex, n_frames - 1, NEW_FRAME_CONCLUSION)
    tex = replace_nth_frame(tex, n_frames, NEW_FRAME_THANKS)

    path.write_text(tex)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
