# Sandbox Service

Sandbox integration service for dynamic behavioral analysis of files and URLs using Cuckoo Sandbox or Any.run.

## Features

- **File Analysis**: Automatic file type detection, hashing, and text extraction
- **Sandbox Integration**: Support for Cuckoo Sandbox and Any.run providers
- **Behavioral Analysis**: Detection of malicious indicators from sandbox results
- **Threat Correlation**: Automatic threat record creation and correlation with detection API
- **Job Queue**: Asynchronous processing using BullMQ and Redis
- **REST API**: Endpoints for file/URL submission and result retrieval

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Sandbox Service                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   File       │  │   Sandbox     │  │   Result     │  │
│  │   Analyzer   │→ │   Submitter  │→ │   Processor  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Behavioral │  │   Correlation│  │   Job Queue  │  │
│  │   Analyzer   │  │   Engine     │  │   Manager    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```env
# Sandbox Provider (cuckoo or anyrun)
SANDBOX_PROVIDER=anyrun

# Cuckoo Sandbox Configuration
CUCKOO_SANDBOX_URL=http://cuckoo-sandbox:8090
CUCKOO_API_KEY=optional-api-key

# Any.run Configuration
ANYRUN_API_KEY=your-api-key

# Sandbox Settings
SANDBOX_TIMEOUT=300
SANDBOX_POLL_INTERVAL=10
SANDBOX_QUEUE_CONCURRENCY=5

# Database & Redis
DATABASE_URL=postgresql://user:pass@host:5432/phishing_detection
REDIS_URL=redis://host:6379

# Detection API (for correlation)
DETECTION_API_URL=http://detection-api:3001
```

## API Endpoints

### Submit File for Analysis

```http
POST /api/v1/sandbox/analyze/file
Content-Type: multipart/form-data

file: <file>
```

**Response:**
```json
{
  "analysis_id": "uuid",
  "status": "pending",
  "message": "File submitted for sandbox analysis"
}
```

### Submit URL for Analysis

```http
POST /api/v1/sandbox/analyze/url
Content-Type: application/json

{
  "url": "https://example.com/suspicious"
}
```

**Response:**
```json
{
  "analysis_id": "uuid",
  "status": "pending",
  "message": "URL submitted for sandbox analysis"
}
```

### Get Analysis Results

```http
GET /api/v1/sandbox/analysis/:id
```

**Response:**
```json
{
  "analysis_id": "uuid",
  "status": "completed",
  "analysis_type": "file",
  "submitted_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:05:00Z",
  "sandbox_provider": "anyrun",
  "sandbox_job_id": "job-id",
  "results": {
    "sandbox": { ... },
    "behavioral": {
      "isMalicious": true,
      "threatScore": 85,
      "indicators": ["c2_communication", "data_exfiltration"]
    },
    "isMalicious": true,
    "threatScore": 85
  },
  "threat_id": "uuid"
}
```

### List Analyses

```http
GET /api/v1/sandbox/analyses?page=1&limit=20
```

## Services

### File Analyzer Service

Analyzes files to determine:
- File type and MIME type
- MD5, SHA1, SHA256 hashes
- Text extraction (PDF, Word documents)
- Whether sandbox analysis is required

### Sandbox Submitter Service

- Submits files/URLs to configured sandbox provider
- Creates database records for tracking
- Handles provider-specific submission logic

### Result Processor Service

- Fetches results from sandbox provider
- Processes behavioral analysis
- Creates threat records for malicious findings
- Updates analysis status

### Behavioral Analyzer Service

Analyzes sandbox results for:
- Suspicious network activity
- C2 communication indicators
- Data exfiltration patterns
- File system modifications
- Process injection attempts
- Malware signatures

### Correlation Service

- Correlates sandbox findings with detection API
- Links sandbox analyses to detection records
- Updates threat confidence scores

### Job Queue

- Asynchronous processing using BullMQ
- Automatic retry with exponential backoff
- Status polling for pending analyses

## Development

### Install Dependencies

```bash
npm install
```

### Build

```bash
npm run build
```

### Run Development Server

```bash
npm run dev
```

### Run Production Server

```bash
npm start
```

## Testing

```bash
npm test
```

## Docker

The service is containerized and can be run with Docker Compose:

```bash
docker-compose up sandbox-service
```

## Integration

The sandbox service integrates with:
- **Detection API**: For correlation of findings
- **Threat Intel Service**: For IOC matching (future enhancement)
- **Database**: PostgreSQL for analysis records and threats
- **Redis**: For job queue management

## License

ISC
