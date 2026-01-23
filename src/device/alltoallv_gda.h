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

	/*if (threadIdx.x == 0) {
		for (int i = 0; i < num_pes; i++) {
			printf("rank = %d, i = %d, count = %zu\n", work->rank, i, work->sendSizes[i]);
		}
	}*/

  	size_t srcOffset = 0;

	/*reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->sendbuff, 1, (void **)&work->sndbuff,
            (work->size*num_pes));*/

	void* dstPtrs[1];
	void* srcPtrs[1];
      //dstPtrs[0] = (void*)(recvbuff + channelOffset);
	for (int i = 0; i < num_pes; i++) {
	    /*srcPtrs[0] = (void*)((char*)work->sendbuff + work->sendDispls[i]);
	    dstPtrs[0] = (void*)((char*)work->sndbuff + srcOffset);*/

	    if (work->sendSizes[i] == 0) {
		if (tid == 0) {
			printf("Problem\n");
		}	
	  	srcOffset + 1024*(i + 1);	    
      		continue;
    	    }

	    ssize_t sendSize =  work->sendSizes[i];
	    char* p = (char*)work->sendbuff + work->sendDispls[i];
	    char* q = (char*)work->sndbuff + srcOffset;

	    reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&p, 1, (void **)&q, sendSize);

	    __syncthreads();

	    void* src1 = (char*)work->sndbuff + srcOffset;
	    void* dst1 = (char*)work->tempbuff + 1024*i;

	    rocshmem::rocshmem_char_put_wg((char*)dst1, (char*)src1, work->sendSizes[i], i);
	    //rocshmem::rocshmem_char_put_nbi_wg((char*)dst1, (char*)src1, work->sendSizes[i], i);

	    srcOffset = srcOffset + 1024*(i + 1);
            //rocshmem::rocshmem_char_alltoall_wg(work->team, ((char*)work->tempbuff), ((char*)work->sndbuff), work->size);
	}

	rocshmem::rocshmem_quiet();
	rocshmem::rocshmem_sync_all_wg();
	//rocshmem::rocshmem_barrier_all();
	//printf("I am back\n");

	//Local copy from symmetric memory to user buffer
	
	for (int i = 0; i < num_pes; i++) {
            /*srcPtrs[0] = (void*)((char*)work->tempbuff + 1024*i);
            dstPtrs[0] = (void*)((char*)work->recvbuff + work->recvDispls[i]);*/

	    char *p = (char*)work->tempbuff + 1024*i;
	    char *q = (char*)work->recvbuff + work->recvDispls[i];

	    ssize_t recvSize =  work->recvSizes[i];
            reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&p, 1, (void **)&q, recvSize);

	    __syncthreads();

	}
    }
 }
};
#endif

