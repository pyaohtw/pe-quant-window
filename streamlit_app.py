import streamlit as st
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

# ---------- helpers ----------
DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def revcomp(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1]

def find_all_occurrences(haystack: str, needle: str) -> List[int]:
    starts = []
    i = haystack.find(needle)
    while i != -1:
        starts.append(i)
        i = haystack.find(needle, i + 1)
    return starts

@dataclass(frozen=True)
class GuideHit:
    guide_name: str
    strand: str          # '+' means guide matches reference; '-' means revcomp(guide) matches reference
    start: int           # 0-based start on reference (leftmost)
    length: int
    match_seq_on_ref: str

    @property
    def end(self) -> int:
        return self.start + self.length - 1

    @property
    def ref_5p(self) -> int:
        # 5' of guide in guide-space mapped to ref axis
        return self.start if self.strand == "+" else self.end

    @property
    def ref_3p(self) -> int:
        # 3' of guide in guide-space mapped to ref axis
        return self.end if self.strand == "+" else self.start

    def nick_index_on_ref(self, nick_offset_from_3p: int) -> int:
        # flip direction on '-' strand
        direction = 1 if self.strand == "+" else -1
        return self.ref_3p + (nick_offset_from_3p * direction)

    def contains_ref_index(self, idx: int) -> bool:
        return self.start <= idx <= self.end


def find_guide_hits(ref: str, guide: str, guide_name: str) -> List[GuideHit]:
    ref_u = ref.upper()
    g_u = guide.upper()
    rc_u = revcomp(g_u)

    hits: List[GuideHit] = []
    for s in find_all_occurrences(ref_u, g_u):
        hits.append(GuideHit(guide_name, "+", s, len(g_u), ref_u[s:s+len(g_u)]))
    for s in find_all_occurrences(ref_u, rc_u):
        hits.append(GuideHit(guide_name, "-", s, len(g_u), ref_u[s:s+len(g_u)]))

    hits.sort(key=lambda h: (h.start, h.strand))
    return hits


def choose_best_pair(peg_hits: List[GuideHit], ng_hits: List[GuideHit]) -> Optional[Tuple[GuideHit, GuideHit]]:
    if not peg_hits or not ng_hits:
        return None

    def score(pair: Tuple[GuideHit, GuideHit]) -> Tuple[int, int]:
        peg, ng = pair
        sep = abs(peg.ref_3p - ng.ref_3p)
        opp = 1 if peg.strand != ng.strand else 0
        return (opp, sep)

    best = None
    best_sc = (-1, -1)
    for p in peg_hits:
        for n in ng_hits:
            sc = score((p, n))
            if sc > best_sc:
                best_sc = sc
                best = (p, n)
    return best


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def compute_window_from_nicks(
    peg_hit: GuideHit,
    ng_hit: GuideHit,
    nick_offset_from_3p: int,
    peg_ext: int,
    ng_ext: int,
    ref_len: int
) -> Tuple[int, int, int, int]:
    """
    Compute quantification window with CRISPResso-correct asymmetry.

    Base rule:
      left boundary  = left_nick  - left_ext
      right boundary = right_nick + (right_ext - 1)

    Additional correction:
      +1 to right boundary for:
        - case 2: peg '+' strand AND peg 5' > ng 5'
        - case 4: peg '-' strand AND peg 5' < ng 5'
    """
    peg_nick = peg_hit.nick_index_on_ref(nick_offset_from_3p)
    ng_nick  = ng_hit.nick_index_on_ref(nick_offset_from_3p)

    # assign left/right by reference coordinate
    if peg_nick <= ng_nick:
        left_nick, right_nick = peg_nick, ng_nick
        left_ext, right_ext = int(peg_ext), int(ng_ext)
    else:
        left_nick, right_nick = ng_nick, peg_nick
        left_ext, right_ext = int(ng_ext), int(peg_ext)

    start = left_nick - left_ext
    end   = right_nick + max(right_ext - 1, 0)

    # ---- CRISPResso asymmetry correction ----
    peg_is_right = peg_hit.ref_5p > ng_hit.ref_5p

    needs_right_plus_one = (
        (peg_hit.strand == "+" and peg_is_right) or
        (peg_hit.strand == "-" and not peg_is_right)
    )

    if needs_right_plus_one:
        end += 1

    # clamp to reference bounds
    start_c = clamp(start, 0, ref_len - 1)
    end_c   = clamp(end,   0, ref_len - 1)
    if start_c > end_c:
        start_c, end_c = end_c, start_c

    return peg_nick, ng_nick, start_c, end_c

def compute_quant_window_center(
    w0: int,
    w1: int,
    peg_hit: GuideHit,
    ng_hit: GuideHit
) -> int:
    """
    Your definition:
      midpoint = floor((w0+w1)/2)
      distance = abs(midpoint - peg_3' end)  (peg_3' end = peg_hit.ref_3p)

    Sign rule:
      - peg '+' : ng5 < peg5 => negative ; ng5 > peg5 => positive
      - peg '-' : ng5 < peg5 => positive ; ng5 > peg5 => negative

    Implemented as:
      base = +1 if ng5 > peg5 else -1
      strand_flip = +1 for peg '+', -1 for peg '-'
      center = distance * base * strand_flip
    """
    midpoint = (w0 + w1) // 2  # floor
    distance = abs(midpoint - peg_hit.ref_3p)

    # if equal (rare), make it 0
    if ng_hit.ref_5p == peg_hit.ref_5p:
        return 0

    base = 1 if (ng_hit.ref_5p > peg_hit.ref_5p) else -1
    strand_flip = 1 if peg_hit.strand == "+" else -1
    return int(distance * base * strand_flip)


def compute_plot_window_size(w0: int, w1: int) -> int:
    """
    plot_window_size is a HALF-window around the center:
      |w1 - w0|/2 + 10
    Rounded up to integer.
    """
    return int(math.ceil(abs(w1 - w0) / 2.0 + 10))


def qc_plot_window_size(
    plot_window_size: int,
    w0: int,
    w1: int,
    ref_len: int,
    peg_hit: GuideHit,
    ng_hit: GuideHit,
    quant_center: int
) -> Tuple[int, Optional[str]]:
    """
    Your QC 4 scenarios reduce to:
      If ng5 > peg5, the plot window extends to the right from peg 3' end:
         peg_3p + |center| + plot_window_size/2 must be <= ref_len-1
      If ng5 < peg5, it extends to the left:
         peg_3p - |center| - plot_window_size/2 must be >= 0

    If it fails, shrink plot_window_size to "size of quantification_window_coordinates"
    which per your plotting convention here is (w1 - w0).
    """
    half = float(plot_window_size)  # already half-window
    max_idx = ref_len - 1
    peg_3p = peg_hit.ref_3p
    center_abs = abs(int(quant_center))

    msg = None
    new_size = plot_window_size

    if ng_hit.ref_5p > peg_hit.ref_5p:
        right_edge = peg_3p + center_abs + half
        if right_edge > max_idx:
            new_size = int(math.ceil(abs(w1 - w0) / 2.0))
            msg = f"QC: plot window would exceed reference right bound (edge≈{right_edge:.1f} > {max_idx}); shrinking --plot_window_size to {new_size}."
    elif ng_hit.ref_5p < peg_hit.ref_5p:
        left_edge = peg_3p - center_abs - half
        if left_edge < 0:
            new_size = int(math.ceil(abs(w1 - w0) / 2.0))
            msg = f"QC: plot window would exceed reference left bound (edge≈{left_edge:.1f} < 0); shrinking --plot_window_size to {new_size}."

    return int(new_size), msg


def primer3_like_view_text(
    ref: str,
    peg: GuideHit,
    ng: GuideHit,
    window_start: int,
    window_end: int,
    line_width: int = 60,
    ng_plus_char: str = "]",
    ng_minus_char: str = "[",
) -> str:
    """
    Primer3-like blocks (no HTML; stable monospace alignment).

    Alignment rule:
      - Put P/N/W labels in the 6-wide index column.
      - Marker strings start exactly under the sequence.

    P line: > (plus), < (minus)
    N line: configurable 1-char markers (default ] and [)
    W line: | boundaries, - interior
    """
    n = len(ref)
    out_lines = []

    def draw_guide(hit: GuideHit, arr_line: List[str], plus_ch: str, minus_ch: str,
                   block_start: int, block_end: int):
        s = max(hit.start, block_start)
        e = min(hit.end, block_end - 1)
        if s > e:
            return
        ch = plus_ch if hit.strand == "+" else minus_ch
        for i in range(s, e + 1):
            arr_line[i - block_start] = ch

    for block_start in range(0, n, line_width):
        block_end = min(n, block_start + line_width)
        seq_block = ref[block_start:block_end]

        peg_line = [" "] * (block_end - block_start)
        ng_line  = [" "] * (block_end - block_start)
        win_line = [" "] * (block_end - block_start)

        draw_guide(peg, peg_line, ">", "<", block_start, block_end)
        draw_guide(ng,  ng_line,  ng_plus_char, ng_minus_char, block_start, block_end)

        if block_start <= window_start < block_end:
            win_line[window_start - block_start] = "|"
        if block_start <= window_end < block_end:
            win_line[window_end - block_start] = "|"

        fill_s = max(window_start, block_start)
        fill_e = min(window_end, block_end - 1)
        for i in range(fill_s, fill_e + 1):
            if win_line[i - block_start] == " ":
                win_line[i - block_start] = "-"

        left_index_1based = block_start + 1
        out_lines.append(f"{left_index_1based:>6} {seq_block}")
        out_lines.append(f"{'P':>6} {''.join(peg_line)}")
        out_lines.append(f"{'N':>6} {''.join(ng_line)}")
        out_lines.append(f"{'W':>6} {''.join(win_line)}")
        out_lines.append("")

    legend = (
        "Legend:\n"
        "  P: pegRNA arrows (>>> on '+' match, <<< on '-' match)\n"
        f"  N: ngRNA  markers ({ng_plus_char*3} on '+' match, {ng_minus_char*3} on '-' match)\n"
        "  W: suggested quantification window (| boundaries, - interior)\n"
    )
    return legend + "\n" + "\n".join(out_lines)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="CRISPResso Quant Window Helper (Prime Editing)", layout="centered")
st.subheader("CRISPResso Prime Editing Window Helper")

st.markdown(
    """
This tool assists in determining CRISPResso2 quantification and plotting window parameters for prime editing experiments.
The quantification window is defined as the region between the pegRNA- and ngRNA-directed nick sites.
"""
)

ref = st.text_area("Reference amplicon sequence (5'→3')", value="AAAAACCCCCTTTTTGGGGG", height=120)
col_peg, col_ng = st.columns(2)
with col_peg:
    peg_seq = st.text_input("pegRNA spacer (5'→3')", value="AACC")
with col_ng:
    ng_seq = st.text_input("ngRNA spacer (5'→3')", value="CCAA")

st.subheader("Nick Position and Quantification Window Options")

c1, c2, c3 = st.columns(3)
with c1:
    nick_offset = st.number_input(
        "Nick position (rel. to gRNA 3′)",
        value=-3,
        step=1,
        help="Negative values indicate upstream positions (SpCas9 default = −3)."
    )
with c2:
    peg_ext = st.number_input(
        "pegRNA window extension (nt)",
        value=10,
        min_value=0,
        step=1,
        help="Number of nucleotides added upstream of the pegRNA nick site for defining the quantification window (default = 10)."
    )
with c3:
    ng_ext = st.number_input(
        "ngRNA window extension (nt)",
        value=10,
        min_value=0,
        step=1,
        help="Number of nucleotides added downstream of the ngRNA nick site for defining the quantification window (default = 10)."
    )
line_width = 60

# marker style option (single-width characters only)
ng_plus_char = "}"
ng_minus_char = "{"


if st.button("Suggest parameters"):
    ref_u = ref.strip().upper().replace(" ", "").replace("\n", "")
    peg_u = peg_seq.strip().upper()
    ng_u  = ng_seq.strip().upper()

    if not ref_u or not peg_u or not ng_u:
        st.error("Please provide reference amplicon, pegRNA, and ngRNA.")
        st.stop()

    for s, name in [(ref_u, "reference"), (peg_u, "pegRNA"), (ng_u, "ngRNA")]:
        bad = set(s) - set("ACGT")
        if bad:
            st.error(f"{name} contains non-ACGT characters: {sorted(bad)}")
            st.stop()

    peg_hits = find_guide_hits(ref_u, peg_u, "pegRNA")
    ng_hits  = find_guide_hits(ref_u, ng_u, "ngRNA")

    if not peg_hits:
        st.error("No exact-match hit for pegRNA (forward or reverse-complement).")
        st.stop()
    if not ng_hits:
        st.error("No exact-match hit for ngRNA (forward or reverse-complement).")
        st.stop()

    pair = choose_best_pair(peg_hits, ng_hits)
    if not pair:
        st.error("Could not choose a peg/ng pair.")
        st.stop()

    peg_hit, ng_hit = pair

    if peg_hit.strand == ng_hit.strand:
        st.warning(
            "pegRNA and ngRNA mapped to the SAME strand. In prime editing they are typically on opposite strands; "
            "please verify your inputs/mapping."
        )

    max_len = max(peg_hit.length, ng_hit.length)
    if abs(int(nick_offset)) > max_len:
        st.error(
            f"Nick offset magnitude (|{nick_offset}|) is greater than at least one guide length "
            f"(peg={peg_hit.length}, ng={ng_hit.length}). Use a smaller offset."
        )
        st.stop()

    peg_nick, ng_nick, w0, w1 = compute_window_from_nicks(
        peg_hit=peg_hit,
        ng_hit=ng_hit,
        nick_offset_from_3p=int(nick_offset),
        peg_ext=int(peg_ext),
        ng_ext=int(ng_ext),
        ref_len=len(ref_u)
    )

    warnings = []
    if not peg_hit.contains_ref_index(peg_nick):
        warnings.append(
            f"pegRNA nick ({peg_nick}) is outside pegRNA span on reference [{peg_hit.start}-{peg_hit.end}]."
        )
    if not ng_hit.contains_ref_index(ng_nick):
        warnings.append(
            f"ngRNA nick ({ng_nick}) is outside ngRNA span on reference [{ng_hit.start}-{ng_hit.end}]."
        )
    # --- new plotting params ---
    quant_center = compute_quant_window_center(w0, w1, peg_hit, ng_hit)
    plot_size = compute_plot_window_size(w0, w1)
    plot_size_qc, qc_msg = qc_plot_window_size(
        plot_window_size=plot_size,
        w0=w0, w1=w1,
        ref_len=len(ref_u),
        peg_hit=peg_hit,
        ng_hit=ng_hit,
        quant_center=quant_center
    )

   # ---- Excel / TSV copy block (values only) ----
    excel_row = f"{w0}-{w1}\t{quant_center}\t{plot_size_qc}"
    st.subheader("Copy for Excel (TSV values only)")
    st.caption(
        "Click the copy icon (top-right of the box) to copy.\n"
        "Paste directly into Excel: columns = quant_window_coordinates | quant_window_center | plot_window_size."
    )
    st.code(excel_row, language="text")
    st.caption("All coordinates are 0-based inclusive (CRISPResso convention).")

    # parameters used for calculation
    st.subheader("Chosen mappings")
    st.write(
        f"pegRNA: strand {peg_hit.strand}, ref_5′={peg_hit.ref_5p}, ref_3′={peg_hit.ref_3p}, "
        f"span=[{peg_hit.start}-{peg_hit.end}], nick={peg_nick}"
    )
    st.write(
        f"ngRNA : strand {ng_hit.strand}, ref_5′={ng_hit.ref_5p}, ref_3′={ng_hit.ref_3p}, "
        f"span=[{ng_hit.start}-{ng_hit.end}], nick={ng_nick}"
    )
    for w in warnings:
        st.warning(w)
    
    # parameters used for CRISPResso
    st.subheader("CRISPResso parameters")
    if qc_msg:
        st.warning(qc_msg)
    params_block = (
        f"--quantification_window_coordinates {w0}-{w1}\n"
        f"--quantification_window_center {quant_center}\n"
        f"--plot_window_size {plot_size_qc}"
    )

    st.code(params_block, language="text")

    st.subheader("Primer3-like visualization")
    view = primer3_like_view_text(
        ref=ref_u,
        peg=peg_hit,
        ng=ng_hit,
        window_start=w0,
        window_end=w1,
        line_width=line_width,
        ng_plus_char=ng_plus_char,
        ng_minus_char=ng_minus_char,
    )
    st.code(view)
