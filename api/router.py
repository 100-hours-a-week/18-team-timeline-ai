from fastapi import APIRouter

# from .timeline import router as timeline_router
# from .merge import router as merge_router
from .hot import router as hot_router

# from .comment import router as comment_router

router = APIRouter()
# router.include_router(timeline_router, prefix="/timeline")
# router.include_router(merge_router, prefix="/merge")
router.include_router(hot_router, prefix="/hot")
# router.include_router(comment_router, prefix="/comment")
