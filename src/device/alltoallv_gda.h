/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include <hip/hip_cooperative_groups.h>

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllToAllvGda, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
        int num_pes = rocshmem::rocshmem_n_pes();

	int numBlocks = gridDim.x;
  	size_t srcOffset = 0;

	work->sendSizes = (size_t*)work->sizes;
	work->sendDispls = (size_t*)work->sizes + num_pes;
    work->recvSizes = (size_t*)work->sizes + 2 * num_pes;
    work->recvDispls = (size_t*)work->sizes + 3 * num_pes;

	if (blockIdx.x >= num_pes)
		return;

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		//printf("NumBlocks = %d, seqNum = %d, nThreads = %d\n", numBlocks, work->flagVal, nThreads);
	}
	int peerIdx = blockIdx.x;
	int i = peerIdx;
	uint64_t val = work->flagVal;
	
	if (work->sendSizes[i] != 0) {

            void* src = (char*)work->sendbuff + work->sendDispls[i];
            void* dst = (char*)work->sndbuff + work->sendDispls[i];
            ssize_t sendSize = work->sendSizes[i];
            reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&src, 1, (void **)&dst,
            sendSize);   
            
            void* src1 = (char*)work->sndbuff + work->sendDispls[i];
            void* dst1 = (char*)work->tempbuff + work->rank*4*1024*1024;
            rocshmem::rocshmem_char_put_nbi_wg((char*)dst1, (char*)src1, work->sendSizes[i], i);
            
	    if (threadIdx.x == 0) { 
	       
	          uint64_t *destFlag = work->flagbuff + work->rank;

	          //void *localFlag = (char*)work->flagbuff + work->rank*num_pes*sizeof(int64_t) + i*sizeof(int64_t);

		  //printf("Val of local flag: bid = %d, i = %d, rank = %d, val = %zu\n", blockIdx.x, i, work->rank, *p);     
	       	  rocshmem::rocshmem_uint64_atomic_set(destFlag, val, i);
	       	  rocshmem::rocshmem_pe_quiet(&i, 1);
                  //printf("Back from wait val = %zu, p = %zu, i = %d, rank = %d\n", val, *p, i, work->rank);
	    }

	    __syncthreads();
        }
	if (work->recvSizes[i] != 0) {
	    if (threadIdx.x == 0) {
	          uint64_t *localFlag = work->flagbuff + i;
	    		    
	       	  rocshmem::rocshmem_uint64_wait_until(localFlag, rocshmem::ROCSHMEM_CMP_EQ, val);
		  //while (*p != val) {
                  /*int ret = rocshmem::rocshmem_uint64_test(localFlag, rocshmem::ROCSHMEM_CMP_EQ, val);
                  if (ret == 0) {
                     printf("problem: expect = %lld, got = %lld\n", val, *localFlag);
                  }*/		
	    }

	    __syncthreads();

	       /*if (threadIdx.x == 0) {
	          printf("I am here bid = %d, rank = %d\n", blockIdx.x, work->rank);
	       }*/

	   void *srcR = (void*)((char*)work->tempbuff + i*4*1024*1024);
           void *dstR = (void*)((char*)work->recvbuff + work->recvDispls[i]);

	   ssize_t recvSize =  work->recvSizes[i];
	   reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
           tid, nThreads, 0, nullptr, false, 1, (void **)&srcR, 1, (void **)&dstR,
           recvSize);
	}
 }
};
#endif

