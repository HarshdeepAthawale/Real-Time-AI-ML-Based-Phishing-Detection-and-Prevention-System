# Setup Scripts

## Initial Setup (First Time Only)

After starting the database for the first time, run this script to create your organization, admin user, and API key:

### Prerequisites
```bash
# 1. Database must be running
docker-compose up -d postgres

# 2. Install dependencies
cd backend/shared/scripts
npm install typescript ts-node yargs bcryptjs @types/yargs @types/bcryptjs typeorm
```

### Run Setup Script

```bash
# From backend/shared/scripts directory
ts-node create-initial-setup.ts \
  --org-name "Your Company Name" \
  --org-domain "yourcompany.com" \
  --admin-email "admin@yourcompany.com" \
  --admin-password "YourSecurePassword123!" \
  --admin-name "Admin User" \
  --tier "enterprise"
```

### Parameters

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--org-name` | Yes | Organization name | - |
| `--org-domain` | Yes | Organization domain | - |
| `--admin-email` | Yes | Admin email address | - |
| `--admin-password` | Yes | Admin password (min 8 chars) | - |
| `--admin-name` | No | Admin full name | "Admin User" |
| `--tier` | No | Subscription tier (free/professional/enterprise) | "enterprise" |

### Example Output

```
🔌 Connecting to database...
✓ Connected

📊 Creating organization...
✓ Organization created: Acme Corp (a1b2c3d4-...)

🔐 Creating admin user...
✓ Admin user created: admin@acme.com (e5f6g7h8-...)

🔑 Creating API key...
✓ API key created: Primary API Key (i9j0k1l2-...)

🛡️  Checking threat intelligence configuration...
✓ Activated 2 threat intelligence feeds:
  - AlienVault OTX
  - URLhaus

======================================================================
🎉 INITIAL SETUP COMPLETE
======================================================================

📋 Your Credentials:
   Organization: Acme Corp
   Admin Email:  admin@acme.com
   Admin Pass:   [provided]

🔑 Your API Key (SAVE THIS - shown only once):
   a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2

📚 Next Steps:
   1. Save your API key securely
   2. Add threat intel API keys to .env
   3. Start all services: docker-compose up -d
   4. Test detection: curl -X POST http://localhost:3001/api/v1/detect/url \
                           -H "X-API-Key: a1b2c3d4..." \
                           -H "Content-Type: application/json" \
                           -d '{"url": "http://example.com"}'

======================================================================
```

## Important Notes

⚠️ **This script should only be run ONCE** for initial setup.

⚠️ **Save your API key** - it's shown only once and cannot be retrieved later.

⚠️ **Use strong passwords** - minimum 8 characters with mix of letters, numbers, symbols.

⚠️ **Threat Intelligence Feeds** - The script will automatically activate feeds if you have configured API keys in your `.env` file:
- `MISP_URL` + `MISP_API_KEY` → Activates MISP
- `OTX_API_KEY` → Activates AlienVault OTX
- `PHISHTANK_API_KEY` → Activates PhishTank
- `VIRUSTOTAL_API_KEY` → Activates VirusTotal
- URLhaus (no key required) → Always activated

## Creating Additional Resources

After initial setup, use the API endpoints to create:

### Additional Organizations
```bash
POST /api/v1/organizations
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json

{
  "name": "New Organization",
  "domain": "neworg.com",
  "subscription_tier": "professional"
}
```

### Additional Users
```bash
POST /api/v1/users
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json

{
  "email": "user@company.com",
  "password": "SecurePassword123!",
  "first_name": "John",
  "last_name": "Doe",
  "role": "analyst"
}
```

### Additional API Keys
```bash
POST /api/v1/api-keys
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json

{
  "name": "Integration API Key",
  "permissions": ["read", "write"]
}
```

## Troubleshooting

### "Organizations already exist"
- This script can only be run once
- Use API endpoints to create additional resources

### "Cannot connect to database"
- Ensure PostgreSQL is running: `docker-compose ps postgres`
- Check DATABASE_URL in .env
- Verify credentials

### "Permission denied"
- Make script executable: `chmod +x create-initial-setup.ts`
- Or use: `ts-node create-initial-setup.ts ...`

### "Module not found"
- Install dependencies: `npm install`
- Use correct working directory: `cd backend/shared/scripts`

## Security Best Practices

1. **Never commit** `.env` files or API keys to git
2. **Rotate API keys** regularly
3. **Use different keys** for dev/staging/production
4. **Store keys securely** (e.g., 1Password, AWS Secrets Manager)
5. **Limit permissions** - only grant what's needed
6. **Monitor usage** - track API key usage for anomalies
7. **Revoke compromised keys** immediately via API
