---
name: Frontend Backend Integration
overview: Integrate frontend and backend with real API calls and workflows. Connect all components to backend services, fix bugs, create missing endpoints, and ensure proper authentication and WebSocket connectivity.
todos:
  - id: fix-threat-dashboard
    content: Fix ThreatDashboard component - correct mapping bug on line 90
    status: completed
  - id: integrate-threat-chart
    content: Integrate ThreatChart component with getChartData() API call
    status: completed
  - id: integrate-threat-intelligence
    content: Integrate ThreatIntelligence component with real API calls for all tabs
    status: completed
  - id: create-intelligence-routes
    content: Create intelligence.routes.ts with domains, patterns, iocs, and summary endpoints
    status: completed
  - id: create-intelligence-service
    content: Create intelligence.service.ts with methods to fetch and format threat intelligence data
    status: completed
  - id: register-intelligence-routes
    content: Register intelligence routes in threat-intel service index.ts
    status: completed
  - id: verify-api-gateway-routing
    content: Verify API Gateway correctly routes /api/v1/intelligence to threat-intel service
    status: completed
  - id: update-env-config
    content: Update environment variables for API and WebSocket URLs
    status: completed
  - id: verify-websocket-config
    content: Verify WebSocket connection URL and event handling
    status: completed
  - id: add-error-handling
    content: Add comprehensive error handling and loading states to all components
    status: completed
---

# Frontend-Backend Integration Plan

## Overview

Integrate the Next.js frontend with all backend services using real API calls. Replace hardcoded data, fix bugs, create missing endpoints, and ensure proper authentication and WebSocket connectivity.

## Current State Analysis

### Frontend Components Status

- **ThreatDashboard**: Uses real APIs (`getDashboardStats`, `getThreatDistribution`) but has a bug mapping over object
- **RecentThreats**: Uses real API (`getRecentThreats`) ✓
- **ThreatChart**: Uses hardcoded data - needs real API integration
- **ThreatIntelligence**: Uses hardcoded data - needs real API integration
- **RealtimeMonitor**: Uses WebSocket hook ✓

### Backend API Status

- **Detection API**: `/api/v1/detect/*`, `/api/v1/dashboard/*` - exists ✓
- **Threat Intelligence**: `/api/v1/ioc/*` - exists but doesn't match frontend expectations
- **Missing**: `/api/v1/intelligence/domains`, `/api/v1/intelligence/patterns`, `/api/v1/intelligence/iocs`, `/api/v1/intelligence/summary`

### Authentication

- Backend requires `X-API-Key` header
- Frontend configured to send from localStorage ✓
- API Gateway also requires API key

## Implementation Tasks

### 1. Fix Frontend Components

#### 1.1 Fix ThreatDashboard Component

**File**: `components/threat-dashboard.tsx`

- Fix bug: Line 90 tries to map over `stats` object instead of `statCards` array
- Ensure proper error handling and loading states

#### 1.2 Integrate ThreatChart with Real API

**File**: `components/threat-chart.tsx`

- Replace hardcoded data with `getChartData()` API call
- Add loading and error states
- Handle empty data gracefully

#### 1.3 Integrate ThreatIntelligence Component

**File**: `components/threat-intelligence.tsx`

- Replace hardcoded data with real API calls:
- `getMaliciousDomains()` for domains tab
- `getThreatPatterns()` for patterns tab
- `getIOCs()` for IOCs tab
- `getThreatIntelligenceSummary()` for summary cards
- Add loading states and error handling
- Handle empty states

### 2. Create Missing Threat Intelligence Endpoints

#### 2.1 Create Intelligence Routes

**File**: `backend/core-services/threat-intel/src/routes/intelligence.routes.ts` (new)

- Create router with endpoints matching frontend expectations:
- `GET /api/v1/intelligence/domains` - Get malicious domains list
- `GET /api/v1/intelligence/patterns` - Get threat patterns
- `GET /api/v1/intelligence/iocs` - Get IOCs (wrapper around IOC search)
- `GET /api/v1/intelligence/summary` - Get intelligence summary stats

#### 2.2 Implement Intelligence Service Methods

**File**: `backend/core-services/threat-intel/src/services/intelligence.service.ts` (new)

- Create service methods that use IOCManagerService:
- `getMaliciousDomains(limit, offset)` - Filter IOCs by type='domain', return formatted list
- `getThreatPatterns(limit)` - Aggregate threat patterns from IOCs
- `getIOCs(limit, offset, type?)` - Wrapper around IOCManager.searchIOCs
- `getSummary()` - Calculate summary statistics

#### 2.3 Register Intelligence Routes

**File**: `backend/core-services/threat-intel/src/index.ts`

- Import and register intelligence routes: `app.use('/api/v1/intelligence', intelligenceRoutes)`

### 3. Update API Gateway Configuration

#### 3.1 Verify Route Mapping

**File**: `backend/api-gateway/src/config/gateway.ts`

- Ensure `/api/v1/intelligence` routes to `threat-intel` service ✓ (already configured)

### 4. Fix Frontend API Client Configuration

#### 4.1 Update API Client Base URL

**File**: `lib/api-client.ts`

- Ensure `NEXT_PUBLIC_API_URL` points to API Gateway (default: `http://localhost:3000`)
- Verify API key header is sent correctly ✓

#### 4.2 Update WebSocket Configuration

**File**: `hooks/use-websocket.ts`

- Ensure `NEXT_PUBLIC_WS_URL` points to correct WebSocket endpoint
- WebSocket should connect to detection-api service (via gateway or directly)
- Default: `ws://localhost:3000` or `ws://localhost:3001` (detection-api port)

### 5. Environment Configuration

#### 5.1 Create/Update Environment Files

**Files**: `.env.local`, `.env.example`

- Add `NEXT_PUBLIC_API_URL=http://localhost:3000` (API Gateway)
- Add `NEXT_PUBLIC_WS_URL=ws://localhost:3000` or direct to detection-api
- Document API key setup in localStorage

### 6. Data Transformation & Type Matching

#### 6.1 Ensure Type Compatibility

**Files**: `lib/types/api.ts`, backend response types

- Verify frontend types match backend responses
- Add transformation functions if needed:
- Map IOC types to frontend `MaliciousDomain` format
- Map IOC data to `ThreatPattern` format
- Map IOC search results to `IOC[]` format
- Calculate summary from IOC stats

### 7. Error Handling & Loading States

#### 7.1 Add Error Boundaries

- Ensure ErrorBoundary wraps components properly
- Add user-friendly error messages
- Handle API errors gracefully

#### 7.2 Loading States

- Add loading skeletons/spinners to all components
- Show loading states during API calls

### 8. Authentication Flow

#### 8.1 API Key Management

**File**: `lib/api-client.ts`

- Verify API key is read from localStorage
- Add API key setup/management UI if needed
- Handle 401 errors (redirect to API key setup)

### 9. WebSocket Integration

#### 9.1 Verify WebSocket Connection

**File**: `hooks/use-websocket.ts`

- Ensure WebSocket connects to correct endpoint
- Handle connection errors
- Verify event types match backend (`threat_detected`, `url_analyzed`, `email_analyzed`)

## File Changes Summary

### New Files

- `backend/core-services/threat-intel/src/routes/intelligence.routes.ts`
- `backend/core-services/threat-intel/src/services/intelligence.service.ts`

### Modified Files

- `components/threat-dashboard.tsx` - Fix mapping bug
- `components/threat-chart.tsx` - Add real API integration
- `components/threat-intelligence.tsx` - Add real API integration
- `backend/core-services/threat-intel/src/index.ts` - Register intelligence routes
- `lib/api-client.ts` - Verify configuration
- `hooks/use-websocket.ts` - Verify WebSocket URL
- `.env.local` / `.env.example` - Add environment variables

## Testing Checklist

- [ ] ThreatDashboard loads real stats and distribution
- [ ] ThreatChart displays real timeline data
- [ ] RecentThreats displays real threats from database
- [ ] ThreatIntelligence displays real domains, patterns, IOCs, and summary
- [ ] WebSocket connects and receives real-time events
- [ ] API key authentication works end-to-end
- [ ] Error states display properly
- [ ] Loading states show during API calls
- [ ] Empty states handle gracefully

## API Endpoint Mapping

| Frontend API Call | Backend Endpoint | Status |
|------------------|------------------|--------|
| `getDashboardStats()` | `GET /api/v1/dashboard/stats` | ✓ Exists |
| `getRecentThreats()` | `GET /api/v1/dashboard/threats` | ✓ Exists |
| `getChartData()` | `GET /api/v1/dashboard/chart` | ✓ Exists |
| `getThreatDistribution()` | `GET /api/v1/dashboard/distribution` | ✓ Exists |
| `getMaliciousDomains()` | `GET /api/v1/intelligence/domains` | ✗ Create |
| `getThreatPatterns()` | `GET /api/v1/intelligence/patterns` | ✗ Create |
| `getIOCs()` | `GET /api/v1/intelligence/iocs` | ✗ Create |
| `getThreatIntelligenceSummary()` | `GET /api/v1/intelligence/summary` | ✗ Create |
| WebSocket events | `ws://localhost:3000` or `ws://localhost:3001` | ✓ Exists |

## Notes

- API Gateway routes `/api/v1/intelligence` to threat-intel service
- Detection API runs on port 3001 (or configured port)
- API Gateway runs on port 3000 (or configured port)
- Frontend should connect to API Gateway for HTTP requests
- WebSocket can connect directly to detection-api or via gateway (if supported)