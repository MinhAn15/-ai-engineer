"""
User Routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db
from api.dependencies import get_current_user
from api.schemas import UserResponse, QuotaResponse, UsageSummary
from api.models import User, Quota, UsageLog

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information.
    """
    return current_user


@router.get("/quota", response_model=QuotaResponse)
async def get_user_quota(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's quota and usage.
    """
    quota = db.query(Quota).filter(Quota.user_id == current_user.id).first()
    
    if not quota:
        # Return unlimited quota
        return QuotaResponse(
            max_tokens_daily=999999999,
            max_documents_daily=999999,
            tokens_used_today=0,
            documents_today=0,
            tokens_remaining=999999999,
            documents_remaining=999999
        )
    
    return QuotaResponse(
        max_tokens_daily=quota.max_tokens_daily,
        max_documents_daily=quota.max_documents_daily,
        tokens_used_today=quota.tokens_used_today,
        documents_today=quota.documents_today,
        tokens_remaining=max(0, quota.max_tokens_daily - quota.tokens_used_today),
        documents_remaining=max(0, quota.max_documents_daily - quota.documents_today)
    )


@router.get("/usage", response_model=UsageSummary)
async def get_usage_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's total usage summary.
    """
    logs = db.query(UsageLog).filter(UsageLog.user_id == current_user.id).all()
    
    total_requests = len(logs)
    total_input = sum(log.input_tokens for log in logs)
    total_output = sum(log.output_tokens for log in logs)
    total_tokens = sum(log.total_tokens for log in logs)
    
    # Gemini pricing estimate
    cost_input = (total_input / 1000) * 0.00025
    cost_output = (total_output / 1000) * 0.0005
    total_cost = cost_input + cost_output
    
    return UsageSummary(
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        estimated_cost_usd=round(total_cost, 4)
    )
