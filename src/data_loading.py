from pathlib import Path

import networkx as nx
import pandas as pd


def load_citations(path: str | Path) -> pd.DataFrame:
    """Load citation edges and normalize paper IDs."""
    citations = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["source", "target"],
        dtype={"source": str, "target": str},
    )
    citations["source"] = citations["source"].str.strip()
    citations["target"] = citations["target"].str.strip()
    return citations


def load_abstracts(root_dir: str | Path) -> pd.DataFrame:
    """Load paper metadata from year-based arXiv abstract directories."""
    root_dir = Path(root_dir)
    metadata: list[dict] = []

    for year_dir in root_dir.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        year = int(year_dir.name)
        for file_path in year_dir.glob("*.abs"):
            paper_id = file_path.stem.strip().lstrip("0")
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            title = next(
                (line.replace("Title: ", "", 1).strip() for line in lines if line.startswith("Title: ")),
                "",
            )
            authors = next(
                (line.replace("Authors: ", "", 1).strip() for line in lines if line.startswith("Authors: ")),
                "",
            )

            separator_count = 0
            abstract_lines: list[str] = []
            for line in lines:
                if line.strip() == "\\\\":
                    separator_count += 1
                    if separator_count == 3:
                        break
                    continue
                if separator_count == 2:
                    abstract_lines.append(line.strip())

            abstract = " ".join(abstract_lines).strip()
            if title and abstract:
                metadata.append(
                    {
                        "paper_id": paper_id,
                        "year": year,
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                    }
                )

    return pd.DataFrame(metadata)


def align_citations_with_metadata(
    citations: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Keep only citation edges whose source and target both exist in metadata."""
    valid_ids = set(metadata["paper_id"])
    return citations[
        citations["source"].isin(valid_ids) & citations["target"].isin(valid_ids)
    ].copy()


def build_citation_graph(citations: pd.DataFrame) -> nx.DiGraph:
    """Create the directed citation graph used by later analysis stages."""
    return nx.from_pandas_edgelist(
        citations,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )
