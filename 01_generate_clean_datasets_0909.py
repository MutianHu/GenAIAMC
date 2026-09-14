"""Stage 1 (0909): create the four independent clean A2G corpora."""

from __future__ import annotations

import argparse

from pipeline_common_0909 import (
    ROLE_SEEDS,
    artifact_summary,
    build_clean_corpus,
    corpus_path,
    save_artifact,
    set_global_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate isolated 0909 clean A2G datasets.")
    parser.add_argument("--roles", nargs="+", choices=sorted(ROLE_SEEDS), default=sorted(ROLE_SEEDS))
    parser.add_argument("--overwrite", action="store_true", help="Replace only the 0909 pipeline's same-role artifact.")
    args = parser.parse_args()
    set_global_seed(2_025)
    for role in args.roles:
        output = corpus_path(role)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite to replace this 0909 artifact.")
        corpus = build_clean_corpus(role)
        save_artifact(corpus, output)
        print(f"saved {role}: {output} {artifact_summary(corpus)}")


if __name__ == "__main__":
    main()
