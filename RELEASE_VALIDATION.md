# Release validation

This release was checked before packaging with the following local tests:

- all Python files parsed successfully with `ast.parse`;
- all Python files compiled successfully with `python -m compileall`;
- no environment-specific user paths or legacy `SD_SEMM` paths were found in the executable Python code;
- the fixed Derm-T2IM and PubMedBERT model references are present;
- both metadata JSON files parse successfully;
- the `whole_all` report list contains seven unique report types;
- retrieval `whole_all` uses seven unique report feature folders, with `reports_shorts` included once;
- synthetic cosine similarity returns `0.0` for zero-norm vectors, as approved for the filtering stage;
- the ZSL cosine helper preserves the original supplied script behavior;
- the experiment-0 smoke-test procedure is documented in `TEST_PROMPT.md`.

A complete GPU end-to-end run cannot be performed without the external datasets, PanDerm repository/checkpoint, Derm-T2IM download, real clinical-note features, and the multimodal checkpoint from the related clinical-note repository. The final runtime validation should therefore be executed in the target research environment using `TEST_PROMPT.md`.
