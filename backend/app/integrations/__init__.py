"""External service integrations.

Each integration owns its own HTTP client, schemas, error mapping and config
guard. Business logic never lives here; the recovery services call into these
adapters, never the other way round.
"""
