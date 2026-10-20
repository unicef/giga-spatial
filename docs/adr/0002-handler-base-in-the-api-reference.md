# Document the handler base in the API reference

The handler base is internal: this repository is its only subclasser, and the hooks are not a stable API for other packages. People adding a dataset still have to see how a source becomes files and how cropping uses a geometry. The API reference documents that once, in a last section named "Handler base." The README, the getting started, and the user guide do not use search key or search geometry. Contributing links to that section and does not explain it again.

## Considered options

- **Leave the base in the contributor notes and out of the API reference.** Rejected: a contributor looks up the classes in the reference, and the notes would be a second copy that drifts.
- **Explain the base on each handler's page.** Rejected: someone loading GHSL would have to read the orchestration.

## Consequences

- Documenting the base does not make it a public extension point. The hooks can still change without a compatibility alias.
- Caller pages show the class a user imports, its constructor, and the methods they call.
