#include "thread_index.h"

#ifdef __INTELLISENSE__
#define CU_INVOKE(__func__) __func__
#define CU_INVOKE1(__func__, __grdSize__, __blkSize__) __func__
#else
#define CU_INVOKE(__func__) __func__<<<grdSize, blkSize>>>
#define CU_INVOKE1(__func__, __grdSize__, __blkSize__) __func__<<<__grdSize__, __blkSize__>>>
#endif

inline unsigned int ceilDiv(unsigned int a, unsigned int b) { return (unsigned int)ceil(a / (float)b); }