# API Reference

## Base URLs
- Financial Analysis: `http://localhost:8001`
- Behavioral Analysis: `http://localhost:8003`

## Authentication
Currently no authentication required (development only).

## Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `symbols` | List[str] | Stock ticker symbols | Required |
| `weights` | List[float] | Portfolio weights (must sum to 1.0) | Equal weight |
| `period` | str | Analysis period: 1month, 3months, 6months, 1year, 2years | 1year |
| `use_real_data` | bool | Use FMP API (true) or synthetic data (false) | true |

## Error Responses

All endpoints return errors in this format:
```json
{
  "success": false,
  "error": "Error message here",
  "error_code": "INVALID_INPUT",
  "timestamp": "2025-09-30T12:00:00Z"
}