# Voice Lock Backend API

A modular, well-organized backend for the Voice Lock authentication system.

## Structure

```
backend/
├── __init__.py                 # Backend package initialization
├── app.py                      # Main FastAPI application
├── requirements.txt            # Backend dependencies
├── README.md                   # This file
├── core/                       # Core services and configuration
│   ├── __init__.py
│   └── services.py             # Service initialization and management
├── database/                   # Database operations
│   ├── __init__.py
│   └── database.py             # SQLite database operations
├── models/                     # Pydantic models and schemas
│   ├── __init__.py
│   └── schemas.py              # Request/response models
├── routes/                     # API route handlers
│   ├── __init__.py
│   ├── auth_routes.py          # Authentication and basic routes
│   ├── voice_routes.py         # Voice processing routes
│   ├── profile_routes.py       # Profile management routes
│   └── security_routes.py      # Security and monitoring routes
└── security/                   # Security and authentication
    ├── __init__.py
    ├── auth.py                 # Authentication utilities
    └── security_manager.py     # Security manager and attack detection
```

## Features

- **Modular Architecture**: Clean separation of concerns with organized modules
- **Voice Authentication**: Complete voice enrollment and verification system
- **Security Features**: Attack detection, rate limiting, and security monitoring
- **Database Integration**: SQLite database with proper schema management
- **API Documentation**: Auto-generated OpenAPI documentation
- **CORS Support**: Cross-origin resource sharing configuration

## API Endpoints

### Authentication & Basic
- `GET /` - Service information
- `GET /health` - Health check

### Voice Processing
- `POST /enroll` - Enroll new voice profile
- `POST /verify` - Verify voice against profile

### Profile Management
- `GET /profiles/{user_id}` - Get voice profile
- `DELETE /profiles/{user_id}` - Delete voice profile

### Security
- `GET /security/events/{user_id}` - Get security events

## Usage

### Running the Backend

```bash
# From the project root
python app_new.py

# Or directly from backend
cd backend
python app.py
```

### Development

The backend is designed to be modular and extensible. Each module has a specific responsibility:

- **Models**: Define data structures and validation
- **Database**: Handle data persistence and retrieval
- **Security**: Manage authentication and attack detection
- **Routes**: Handle HTTP requests and responses
- **Core**: Initialize and manage services

### Adding New Features

1. **New Models**: Add to `models/schemas.py`
2. **New Routes**: Create new router in `routes/` and include in `app.py`
3. **New Database Operations**: Add methods to `database/database.py`
4. **New Security Features**: Extend `security/security_manager.py`

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Pydantic: Data validation
- SQLite3: Database (built-in)

## Configuration

The backend uses a service-based architecture where all core services are initialized once and shared across the application. This ensures consistency and efficient resource usage.
