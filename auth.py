from jose import JWTError, jwt
from datetime import datetime, timedelta


SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

fake_user_db = {
    "admin": {
        "username": "admin",
        "password": "admin123"
    }
}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_user(username: str, password: str):
    user = fake_user_db.get(username)
    if not user or user["password"] != password:
        return None
    return user