# A TifProcessor reads one working file and derives new processors

A `TifProcessor` reads exactly one working file, a stored file or a derived file, through one internal owner. `open_dataset`, worker setup, the memory estimate, and the view generators' merge of a processor list all ask that owner. A stored file in a non-local data store is copied once into a temp dir owned by the instance that needs it, so every open reads a local file. Merging, reprojecting, and clipping go through one internal derivation path. It returns a new processor that owns its derived file and carries its parent's settings and lineage. `to_crs` replaces `reproject_to`.

## Considered options

- **Stream non-local stored files through GDAL VSI or a file object.** Rejected for now: no second data store needs it, and a local copy also lets worker processes open the file by path.
- **Keep reading whole stored files into memory on every open.** Rejected: every call downloaded the file again, and every batched-sampling worker held a full copy in memory.
- **Deprecate `reproject_to` before removing it.** Rejected: the library is before 1.0 ([ADR 0003](0003-public-import-paths.md)), nothing in it calls `reproject_to`, and its `output_path` wrote to the local filesystem except when the CRS already matched.
- **A public merge of processors.** Rejected: the view generators are its only caller.
- **Restore `dataset_path` to the caller's input.** Rejected: it would change a public field's meaning, and nothing internal needs it.

## Consequences

- `BaseViewGenerator` is the only module outside `TifProcessor` that may call its private processor merge. Public generator constructors and user code use public methods only.
- `dataset_path` still names a derived file after a merge, reprojection, or clip. Internal code does not read it.
- A copy of a processor never deletes files it did not create. A processor over a stored file stays readable after `cleanup()`, and a processor over a derived file does not.
- `is_merged` and `source_count` describe the original stored files on every derived processor.
