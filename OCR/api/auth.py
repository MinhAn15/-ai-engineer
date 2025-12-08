"""
Authentication module with JWT and password hashing
"""
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import os

from jose import JWTError, jwt
from sqlalchemy.orm import Session

from api.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from api.models import User
from api.schemas import TokenData


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash (SHA256 with salt)"""
    # Format: salt$hash
    if '$' not in hashed_password:
        return False
    salt, stored_hash = hashed_password.split('$', 1)
    computed_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
    return computed_hash == stored_hash


def get_password_hash(password: str) -> str:
    """Hash a password with salt using SHA256"""
    salt = os.urandom(16).hex()
    password_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${password_hash}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode JWT token and extract data"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if username is None:
            return None
        
        return TokenData(username=username, user_id=user_id)
    except JWTError:
        return None


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user


def create_user(
    db: Session, 
    email: str, 
    username: str, 
    password: str,
    is_admin: bool = False
) -> User:
    """Create a new user (admin only)"""
    hashed_password = get_password_hash(password)
    
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        is_admin=is_admin
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create default quota for user
    from api.models import Quota
    from api.config import DEFAULT_DAILY_TOKEN_LIMIT, ADMIN_DAILY_TOKEN_LIMIT, DEFAULT_DAILY_DOCUMENT_LIMIT
    
    # Admin gets higher quota
    token_limit = ADMIN_DAILY_TOKEN_LIMIT if is_admin else DEFAULT_DAILY_TOKEN_LIMIT
    
    quota = Quota(
        user_id=user.id,
        max_tokens_daily=token_limit,
        max_documents_daily=DEFAULT_DAILY_DOCUMENT_LIMIT
    )
    db.add(quota)
    db.commit()
    
    return user
