from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.customer import Customer
from app.models.subscription import Subscription
from app.schemas.subscription import (
    SubscriptionCreate,
    SubscriptionListResponse,
    SubscriptionResponse,
)

router = APIRouter(
    prefix="/api/v1/subscriptions",
    tags=["subscriptions"],
)


@router.post(
    "/",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_subscription(
    payload: SubscriptionCreate,
    db: Session = Depends(get_db),
) -> Subscription:
    customer = db.get(Customer, payload.customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    subscription = Subscription(**payload.model_dump())
    db.add(subscription)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Subscription with this external_subscription_id already exists.",
        )
    db.refresh(subscription)
    return subscription


@router.get(
    "/",
    response_model=list[SubscriptionListResponse],
    status_code=status.HTTP_200_OK,
)
def list_subscriptions(db: Session = Depends(get_db)) -> list[Subscription]:
    return list(db.scalars(select(Subscription)).all())


@router.get(
    "/{subscription_id}",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_200_OK,
)
def get_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
) -> Subscription:
    subscription = db.get(Subscription, subscription_id)
    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found.",
        )
    return subscription
