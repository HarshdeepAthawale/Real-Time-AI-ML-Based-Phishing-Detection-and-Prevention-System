-- Initial schema migration
-- This file combines all schema files in the correct order

-- Run all schema files in dependency order
\i schemas/users.sql
\i schemas/threats.sql
\i schemas/domains.sql
\i schemas/ml_models.sql
\i schemas/threat_intel.sql
\i schemas/emails.sql
\i schemas/sandbox.sql
