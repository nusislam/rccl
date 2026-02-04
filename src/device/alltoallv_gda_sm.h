/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllToAllvGdaSm, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
    //if (blockIdx.x == 0) {
        int num_pes = rocshmem::rocshmem_n_pes();
	int numBlocks = gridDim.x;
  	size_t srcOffset = 0;
	ssize_t recvSize;

	int sizePerBlock = (work->size)/numBlocks;

	void *src = (T*)work->sendbuff + blockIdx.x * sizePerBlock;
        void *dst = (T*)work->sndbuff + blockIdx.x * sizePerBlock;

	reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&src, 1, (void **)&dst,
            sizePerBlock);

	/*reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->sizes, 1, (void **)&work->sendSizes,
            (ssize_t)(num_pes*sizeof(size_t)));*/

	if (blockIdx.x == 0) {
	   work->sendSizes = (size_t*)work->sizes;
	   work->sendDispls = (size_t*)work->sizes + num_pes;
           work->recvSizes = (size_t*)work->sizes + 2 * num_pes;
           work->recvDispls = (size_t*)work->sizes + 3 * num_pes;

	   rocshmem::rocshmem_char_alltoallv_wg(work->team, (char*)work->tempbuff, work->recvSizes, work->recvDispls, 
			(char*)work->sndbuff, work->sendSizes, work->sendDispls);

	   recvSize = work->recvDispls[num_pes - 1] + work->recvSizes[num_pes - 1];
	   sizePerBlock = recvSize/numBlocks;
	}

	void *srcR = (T*)work->tempbuff + blockIdx.x * sizePerBlock;
        void *dstR = (T*)work->recvbuff + blockIdx.x * sizePerBlock;

	reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&srcR, 1, (void **)&dstR,
            sizePerBlock);

    //}
 }
};
#endif

