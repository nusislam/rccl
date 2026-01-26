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

	void* dstPtrs[1];
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

	    //rocshmem::rocshmem_char_put_wg((char*)dst1, (char*)src1, work->sendSizes[i], i);
	    rocshmem::rocshmem_char_put_nbi_wg((char*)dst1, (char*)src1, work->sendSizes[i], i);

	}

	//rocshmem::rocshmem_quiet();
	rocshmem::rocshmem_sync_all_wg();

	//Local copy from symmetric memory to user buffer
	
	for (int i = 0; i < num_pes; i++) {
            srcPtrs[0] = (void*)((char*)work->tempbuff + work->rank*1024 + i*512);
            dstPtrs[0] = (void*)((char*)work->recvbuff + work->recvDispls[i]);

	    ssize_t recvSize =  work->recvSizes[i];

	    reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, srcPtrs, 1, dstPtrs, recvSize);
	}
    }
 }
};
#endif

