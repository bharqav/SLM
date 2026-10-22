from fastapi import APIRouter

from backend.api.v1 import admin, auth, chat, conversations, feedback, gdpr, legal, meta, oauth_routes

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(oauth_routes.router)
api_router.include_router(legal.router)
api_router.include_router(meta.router)
api_router.include_router(conversations.router)
api_router.include_router(chat.router)
api_router.include_router(feedback.router)
api_router.include_router(admin.router)
api_router.include_router(gdpr.router)
