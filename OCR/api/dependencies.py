"""
FastAPI Dependencies
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from api.database import get_db
from api.auth import decode_token
from api.models import User

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require admin privileges"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


def check_quota(user: User, db: Session, tokens_needed: int = 0) -> bool:
    """Check if user has remaining quota"""
    from datetime import datetime, timedelta
    from api.models import Quota
    
    quota = db.query(Quota).filter(Quota.user_id == user.id).first()
    if not quota:
        return True  # No quota means unlimited
    
    # Reset daily quota if needed
    now = datetime.utcnow()
    if quota.last_reset.date() < now.date():
        quota.tokens_used_today = 0
        quota.documents_today = 0
        quota.last_reset = now
        db.commit()
    
    # Check limits
    if quota.tokens_used_today + tokens_needed > quota.max_tokens_daily:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily token limit exceeded ({quota.max_tokens_daily:,} tokens/day)"
        )
    
    if quota.documents_today >= quota.max_documents_daily:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily document limit exceeded ({quota.max_documents_daily} documents/day)"
        )
    
    return True


def update_quota_usage(user: User, db: Session, tokens_used: int, document_count: int = 0):
    """Update user's quota usage"""
    from api.models import Quota
    
    quota = db.query(Quota).filter(Quota.user_id == user.id).first()
    if quota:
        quota.tokens_used_today += tokens_used
        quota.documents_today += document_count
        db.commit()
