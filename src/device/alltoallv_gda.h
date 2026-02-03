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
struct RunWorkColl<ncclFuncAllToAllvGda, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
    if (blockIdx.x == 0) {
        int num_pes = rocshmem::rocshmem_n_pes();

  	size_t srcOffset = 0;

	reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->sendbuff, 1, (void **)&work->sndbuff,
            (work->size));

	reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->sizes, 1, (void **)&work->sendSizes,
            (ssize_t)(num_pes*sizeof(size_t)));

	void* dstPtrs[1];
        void* srcPtrs[1];

	/*for (int i = 0; i < num_pes; i++) {
		printf("i = %d, sendSize = %zu\n", i, work->sendSizes[i]);
	}*/
	/*srcPtrs[0] = (void*)((size_t*)work->sizes + num_pes);
        dstPtrs[0] = (void*)((size_t*)work->sendDispls);

	reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, srcPtrs, 1, dstPtrs,
            (ssize_t)(num_pes*sizeof(size_t)));

	srcPtrs[0] = (void*)((size_t*)work->sizes + 2*num_pes);
        dstPtrs[0] = (void*)((size_t*)work->recvSizes);

        reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, srcPtrs, 1, dstPtrs,
            (ssize_t)(num_pes*sizeof(size_t)));


	srcPtrs[0] = (void*)((size_t*)work->sizes + 3*num_pes);
        dstPtrs[0] = (void*)((size_t*)work->recvDispls);

        reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, srcPtrs, 1, dstPtrs,
            (ssize_t)(num_pes*sizeof(size_t)));*/

	work->sendDispls = (size_t*)work->sizes + num_pes;
        work->recvSizes = (size_t*)work->sizes + 2 * num_pes;
        work->recvDispls = (size_t*)work->sizes + 3 * num_pes;

	rocshmem::rocshmem_char_alltoallv_wg(work->team, (char*)work->tempbuff, work->recvSizes, work->recvDispls, 
			(char*)work->sndbuff, work->sendSizes, work->sendDispls);

	ssize_t recvSize = work->recvDispls[num_pes - 1] + work->recvSizes[num_pes - 1];

	/*if (tid == 0) {
		printf("sendSize = %zu, recvSize = %zu\n", work->size, recvSize);
	}*/

	reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->tempbuff, 1, (void **)&work->recvbuff,
            recvSize);


	/*void* dstPtrs[1];
	void* srcPtrs[1];
	for (int i = 0; i < num_pes; i++) {

	    if (work->sendSizes[i] == 0) {
		if (tid == 0) {
			printf("Problem\n");
		}	
      		continue;
    	    }

	    void* src1 = (char*)work->sndbuff + work->sendDispls[i];
	    void* dst1 = (char*)work->tempbuff + 1024*i + work->rank*512;

	    rocshmem::rocshmem_char_put_nbi_wg((char*)dst1, (char*)src1, work->sendSizes[i], i);
	}

	rocshmem::rocshmem_sync_all_wg();

	//Local copy from symmetric memory to user buffer
	
	for (int i = 0; i < num_pes; i++) {
            srcPtrs[0] = (void*)((char*)work->tempbuff + work->rank*1024 + i*512);
            dstPtrs[0] = (void*)((char*)work->recvbuff + work->recvDispls[i]);

	    ssize_t recvSize =  work->recvSizes[i];

	    reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, srcPtrs, 1, dstPtrs, recvSize);
	}*/
    }
 }
};
#endif

