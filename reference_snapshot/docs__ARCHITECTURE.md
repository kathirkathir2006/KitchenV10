# Architecture — KitchenIQ-OS

**Authoritative product/engine specification:** [`KitchenIQ_Architecture_Specification_v9.5.md`](./KitchenIQ_Architecture_Specification_v9.5.md) (V9.5 / Document Revision 31).

That document supersedes the former Cook Engine Architecture v9.4 title for product architecture, security, cook pipeline, vision, and V9.5 predictive household food system requirements.

## Local engineering layout (unchanged)

1. Intelligence lives in the backend (cook domain, food vision, registry, security) — not in Flutter.
2. Flutter is UI + HTTP client only.
3. If backend is unreachable → fail closed / maintenance — no fake intelligence.
4. Product APIs: `/api/v1/...`

See the authoritative V9.5 specification for north star, KitchenState singularity, Safety/Policy authority, Trajectory/Twin/Use First (SPECIFIED), and preserved v9.4 contracts.
